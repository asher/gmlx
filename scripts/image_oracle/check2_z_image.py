#!/usr/bin/env python3
"""Check 2 oracle for Z-Image: the text encoder tap.

The pinned pipeline's encode_prompt runs Qwen3-4B, loaded from its GGUF by
transformers on the CPU, on each prompt in the fixed chat template with
thinking on, padded to 512 tokens and masked. Its caption features are
hidden_states[-2], which this script checks equal, bit for bit, to the
output of decoder layer 34 read by a forward hook: the state after 35 of
36 layers, before the final norm. It also checks that hidden_states[-1]
is the final norm of the last layer's output.

For each prompt the fixture holds the token ids and the fp32 and bf16 taps
of the real rows. The manifest records the bf16 against fp32 gap of each
prompt, the gap between the padded batch and an unpadded batch 1 run, and
the tolerances: relative L2 at twice the bf16 gap, per-token cosine at its
starting value, 0.999.

The bf16 run computes each linear product as a float32 GEMM over the
bfloat16 operands, rounded once to bfloat16, since torch's CPU bfloat16
GEMM is single threaded. That differs from a native bfloat16 GEMM with
float32 accumulation only in summation order. The bf16 copy keeps its
rotary frequencies in fp32, as a model loaded in bf16 keeps them.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_z_image.py --encoder Qwen3-4B-Q4_K_M.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

NAME = "check2_z_image"
MAX_TOKENS = 512
TAP_LAYER = 34          # zero-based index of the 35th decoder layer
STARTING = {"rel_l2": 2e-2, "token_cosine_min": 0.999}
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt; per-token cosine at the starting value")

# Text that never repeats itself. A prompt made of one paragraph repeated
# gives near ties in attention to the earlier copies, and the oracle's own
# bf16 run then leaves single tokens at a cosine of 0.3.
_LONG = (
    "A narrow cobblestone street in an old harbor town at dusk, lined with "
    "three story houses painted ochre, teal and faded red. Laundry hangs on "
    "lines strung between the balconies, and a bicycle with a wicker basket "
    "leans against a green door. A fishmonger closes the shutters of a "
    "small shop while gulls circle above the masts in the harbor. Warm "
    "lamplight spills from a cafe where two old men play chess at a "
    "marble table, and a cat watches them from a windowsill full of "
    "geraniums. Puddles from an afternoon rain reflect the lamps and the "
    "violet sky, and the far end of the street opens onto the sea, where a "
    "lighthouse is just switching on. In the foreground a girl in a yellow "
    "raincoat crouches to inspect a snail on the curb, her boots splashed "
    "with mud. Behind her a baker carries a tray of round loaves dusted "
    "with flour toward a delivery van whose rear doors stand open. A "
    "hand painted sign above the bakery reads Panificio in chipped gold "
    "letters. Strings of small bulbs zigzag overhead, some of them dark, "
    "and a pigeon sits on the wire between two of them. On the left, a "
    "stone staircase climbs toward a chapel with a bell tower, its steps "
    "worn into shallow bowls by centuries of feet. Moss grows in the "
    "cracks, and someone has left a bunch of wildflowers on the third step. "
    "A scooter with a dented fender is parked at an angle beside a bollard "
    "shaped like a lion. The sky shades from deep indigo at the top of the "
    "frame to a band of apricot just above the horizon, with a thin "
    "crescent moon and a single bright planet. Fishing boats with blue and "
    "white hulls rock at their moorings, nets heaped on their decks, and "
    "a man in a wool cap coils rope on the quay. Far out on the water the "
    "running lights of a ferry move slowly toward an island whose hills "
    "are already black. The air looks cool and damp, with a faint haze "
    "that softens the distant lights into halos. Shot on medium format "
    "film with a normal lens at eye level, shallow depth of field on the "
    "girl and the snail, gentle grain, muted greens and warm highlights, "
    "no text other than the bakery sign, no watermark. The composition "
    "follows the curve of the street from the lower left corner to the "
    "lighthouse at the upper right, leading the eye past each small "
    "story along the way. Every window tells something: a family eating "
    "soup under a hanging lamp, a tailor pinning the hem of a coat on a "
    "wooden dummy, an empty room with a piano and a vase of dry roses, a "
    "boy practicing the violin with his music stand turned toward the "
    "glass. Rainwater still drips from the gutters, catching the light in "
    "bright beads, and the cobbles shine like the backs of beetles. A "
    "black dog trots out of an alley with a paper bag in its mouth, "
    "looking over its shoulder as if someone might follow.")

PROMPTS = {
    "plain": ("A red bicycle leaning against a weathered brick wall at "
              "sunset, a cat asleep in its basket, warm light, 35mm "
              "photograph"),
    "empty": "",
    "non_ascii": ("Un caf\u00e9 \u00e0 Montr\u00e9al en hiver, "
                  "\u7a93\u8fba\u3067\u5bdd\u3066\u3044\u308b\u732b, "
                  "\u4e00\u676f\u62ff\u94c1\u548c\u4e00\u672c\u4e66, "
                  "\u0441\u043d\u0435\u0433 \u0437\u0430 \u043e\u043a\u043d"
                  "\u043e\u043c, \U0001f431\u2615"),
    "long": _LONG,
}


def emulate_bf16_linear(torch, enable: bool) -> None:
    """Swap F.linear for a float32 GEMM over bfloat16 operands with one
    bfloat16 rounding of the result, or restore it."""
    import torch.nn.functional as F

    native = getattr(F, "_check2_native_linear", None)
    if native is None:
        native = F._check2_native_linear = F.linear
    if not enable:
        F.linear = native
        return

    def linear(x, w, b=None):
        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            y = native(x.float(), w.float(), None if b is None else b.float())
            return y.to(torch.bfloat16)
        return native(x, w, b)

    F.linear = linear


def load_encoder(torch, a):
    from transformers import AutoConfig, Qwen3Model

    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    # Without a device_map, transformers places GGUF weights on MPS when
    # torch has it. The oracle computes on the CPU.
    te = Qwen3Model.from_pretrained(
        str(a.encoder.parent), gguf_file=a.encoder.name, config=cfg,
        torch_dtype=torch.float32, device_map="cpu",
        local_files_only=True).eval()
    te16 = copy.deepcopy(te).to(torch.bfloat16).eval()
    # A model loaded in bf16 keeps its rotary frequencies in fp32, which
    # the cast rounds.
    te16.rotary_emb.inv_freq = te.rotary_emb.inv_freq.clone()
    for m, want in ((te, torch.float32), (te16, torch.bfloat16)):
        where = {str(p.device) for p in m.parameters()}
        kinds = {p.dtype for p in m.parameters()}
        if where != {"cpu"} or kinds != {want}:
            raise SystemExit(f"error: text encoder on {where} in {kinds}")
    if len(te.layers) != 36:
        raise SystemExit(f"error: {len(te.layers)} layers, expected 36")
    return te, te16


class Tap:
    """Outputs of decoder layers TAP_LAYER and the last one."""

    def __init__(self, model):
        self.out = {}
        last = len(model.layers) - 1
        self._hooks = [model.layers[i].register_forward_hook(self._hook(i))
                       for i in (TAP_LAYER, last)]

    def _hook(self, i):
        def hook(_mod, _args, out):
            self.out[i] = out[0] if isinstance(out, tuple) else out
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


def encode(torch, pipe, model, prompt: str) -> dict:
    """encode_prompt on ``model`` with the layer hooks, and the checks that
    tie its output to the hooked tap."""
    pipe.text_encoder = model
    tap = Tap(model)
    try:
        t0 = time.perf_counter()
        (embeds,) = pipe.encode_prompt(prompt, device=torch.device("cpu"),
                                       do_classifier_free_guidance=False)[0]
        secs = time.perf_counter() - t0
    finally:
        tap.remove()
    text = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    toks = pipe.tokenizer(text, padding="max_length", max_length=MAX_TOKENS,
                          truncation=True, return_tensors="pt")
    mask = toks.attention_mask[0].bool()
    hooked = tap.out[TAP_LAYER][0][mask]
    last = len(model.layers) - 1
    # hidden_states[-1] as Qwen3Model returns it: the final norm of the last
    # layer's output.
    normed = model(input_ids=toks.input_ids,
                   attention_mask=toks.attention_mask).last_hidden_state
    return {
        "text": text, "ids": toks.input_ids[0][mask].numpy().astype(np.int64),
        "tap": oracle.to_numpy(embeds), "seconds": secs,
        "hook_equals_hidden_states_minus_2": bool(torch.equal(hooked,
                                                              embeds)),
        "last_is_final_norm": bool(torch.equal(
            model.norm(tap.out[last])[0][mask], normed[0][mask])),
    }


def unpadded(torch, model, ids: np.ndarray) -> np.ndarray:
    """The tap of one prompt at batch 1 with no padding and no mask."""
    tap = Tap(model)
    try:
        model(input_ids=torch.from_numpy(ids)[None])
    finally:
        tap.remove()
    return oracle.to_numpy(tap.out[TAP_LAYER][0])


def token_cosine(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    return (g * r).sum(-1) / (np.linalg.norm(g, axis=-1)
                              * np.linalg.norm(r, axis=-1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-4B text GGUF")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo (tokenizer "
                         "and text_encoder config)")
    ap.add_argument("--config-revision",
                    default="f332072aa78be7aecdf3ee76d5c247082da564a6",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()

    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(a.config / "tokenizer"),
                                        local_files_only=True)
    t0 = time.perf_counter()
    te, te16 = load_encoder(torch, a)
    load_s = time.perf_counter() - t0
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    pipe = ZImagePipeline(scheduler=sched, vae=None, text_encoder=te,
                          tokenizer=tok, transformer=None)

    fx = fixtures.FixtureWriter(
        NAME, check=2, family="z_image",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--encoder", a.encoder.name, "--config",
              a.config.name])
    fx.add_file("encoder", a.encoder)
    fx.add_file("chat_template", a.config / "tokenizer"
                / "tokenizer_config.json")
    gaps, pads, checks, rendered, lengths, secs = {}, {}, {}, {}, {}, {}
    tol_rel = {}
    for key, prompt in PROMPTS.items():
        emulate_bf16_linear(torch, False)
        r32 = encode(torch, pipe, te, prompt)
        flat = unpadded(torch, te, r32["ids"])
        emulate_bf16_linear(torch, True)
        r16 = encode(torch, pipe, te16, prompt)
        emulate_bf16_linear(torch, False)
        if not np.array_equal(r16["ids"], r32["ids"]):
            raise SystemExit(f"error: {key}: ids differ between modes")
        if r32["tap"].shape != (len(r32["ids"]), 2560):
            raise SystemExit(f"error: {key}: tap shape {r32['tap'].shape}")
        fx.add(f"ids_{key}", r32["ids"])
        fx.add(f"tap_{key}", r32["tap"])
        fx.add(f"tap_bf16_{key}", r16["tap"])
        gap = fixtures.compare(r16["tap"], r32["tap"])
        gap["token_cosine_min"] = float(token_cosine(r16["tap"],
                                                     r32["tap"]).min())
        gaps[key] = gap
        pad = fixtures.compare(flat, r32["tap"])
        pad["token_cosine_min"] = float(token_cosine(flat, r32["tap"]).min())
        pads[key] = pad
        checks[key] = {m: {k: r[k] for k in (
            "hook_equals_hidden_states_minus_2", "last_is_final_norm")}
            for m, r in (("fp32", r32), ("bf16", r16))}
        rendered[key] = r32["text"]
        lengths[key] = len(r32["ids"])
        secs[key] = {"fp32_s": round(r32["seconds"], 1),
                     "bf16_s": round(r16["seconds"], 1)}
        tol_rel[key] = 2 * gap["rel_l2"]
        print(f"[check2] {key}: {lengths[key]} tokens, bf16 vs fp32 rel L2 "
              f"{gap['rel_l2']:.3e}, token cos min "
              f"{gap['token_cosine_min']:.6f}, unpadded vs padded rel L2 "
              f"{pad['rel_l2']:.3e}", flush=True)
        if not all(all(v.values()) for v in checks[key].values()):
            raise SystemExit(f"error: {key}: tap checks failed: "
                             f"{checks[key]}")

    if lengths["long"] != MAX_TOKENS:
        raise SystemExit(f"error: the long prompt has {lengths['long']} "
                         f"tokens, not the {MAX_TOKENS} cap")
    fx.inputs.update(
        prompts=PROMPTS, rendered=rendered, lengths=lengths,
        max_sequence_length=MAX_TOKENS, tap="output of decoder layer 34 "
        "(35 of 36 layers run), before the final norm",
        template="tokenizer.apply_chat_template, add_generation_prompt=True,"
        " enable_thinking=True", padding="max_length 512, right, masked",
        config_revision=a.config_revision,
        encoder="transformers Qwen3Model from GGUF through its legacy loader,"
        " gguf-py float32 dequantization; the bf16 model is a bfloat16 copy "
        "of the float32 one, with its rotary frequencies left in float32")
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                "rounding"},
        tap_checks=checks, bf16_vs_fp32=gaps, unpadded_vs_padded=pads,
        seconds={"load_s": round(load_s, 1), **secs})
    fx.tolerances.update(rule=RULE, starting=STARTING, rel_l2=tol_rel,
                         token_cosine_min=STARTING["token_cosine_min"])
    fx.note("The unpadded rows run at batch 1 with no mask, as gmlx encodes "
            "a prompt. Their gap to the padded reference is rounding only, "
            "since a causal layer never reads a later pad.")
    path = fx.write(a.out)
    print(f"[check2] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
