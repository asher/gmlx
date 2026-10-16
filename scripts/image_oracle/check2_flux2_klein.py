#!/usr/bin/env python3
"""Check 2 oracle for FLUX.2 klein: the text encoder tap.

BFL's Qwen3Embedder.forward (src/flux2/text_encoder.py of --reference)
renders each prompt in the Qwen3 chat template of one user turn with the
generation prompt and enable_thinking False, tokenizes it with padding to
max_length 512 and truncation there, and runs Qwen3-4B with the attention
mask and no position ids, so each row takes its index as its position and
each pad row reads the real tokens alone. It stacks hidden_states 9, 18
and 27 of every row, the pad rows kept, to (512, 3 * 2560), layer by
layer, in the model's dtype. This script calls that method itself, on an
instance whose model is the GGUF's and whose tokenizer is the official
folder's, and checks that hidden_states[k] is the output of decoder layer
k - 1, read by a forward hook: the state after k layers, before the final
norm.

transformers builds the model as Qwen3ForCausalLM from the config of the
official text_encoder folder, and gguf-py dequantizes each tensor of the
GGUF to float32 into it by name. The output matrix is left out: the tap
never reads it. The bf16 run casts that model to bf16 and keeps its
rotary frequencies in fp32, as a model loaded in bf16 keeps them.

For each prompt the fixture holds the rendered text, the ids and mask the
method fed the model, the rotary positions the model took, and the fp32
and bf16 features of every row. The manifest records the bf16 against
fp32 gap of each prompt over all rows, the real rows and the pad rows,
the smallest and largest relative L2 of a pad row of the fp32 run to the
first pad row, and the tolerances: relative L2 at twice each gap, and a
per-token cosine floor of 1 - 4 (1 - c) on the real rows, c the oracle's
own smallest per-token cosine of the real rows of the prompt.

The pad rows take no cosine floor. In the oracle's own bf16 run, a change
of the arithmetic at the size of one bf16 rounding moves the pad rows by
about a tenth in relative L2 and single pad rows to a cosine near 0.5,
where the real rows move by about 2e-3. A floor from that run would hold
nothing.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_flux2_klein.py --reference <flux2>/src \\
      --reference-commit <sha> --encoder Qwen3-4B-Q8_0.gguf \\
      --text-encoder <official repo>/text_encoder \\
      --tokenizer <official repo>/tokenizer
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

import fixtures
import oracle
from check2_qwen_image21 import HEAD, dequantized, hf_name
from check2_z_image import _LONG, PROMPTS, token_cosine

NAME = "check2_flux2_klein"
SCRIPT = "scripts/image_oracle/check2_flux2_klein.py"
RULE = ("relative L2 = 2 x the oracle's own bf16 against fp32 relative L2 "
        "of the same prompt, over all rows, over the real rows and over "
        "the pad rows; per-token cosine of each real row at least "
        "1 - 4 (1 - c), c the oracle's own smallest per-token cosine of the "
        "real rows of the same prompt")
# The 4B widths, from the GGUF header.
WIDTHS = {"hidden_size": "embedding_length",
          "intermediate_size": "feed_forward_length",
          "num_hidden_layers": "block_count",
          "num_attention_heads": "attention.head_count",
          "num_key_value_heads": "attention.head_count_kv",
          "head_dim": "attention.key_length"}


def reference(path: Path):
    """The reference's text_encoder module. It imports sampling, which
    imports torchvision for its image preparation, which the tap does not
    use and the oracle environment lacks, so an empty module stands in
    while it imports. transformers looks torchvision up by its spec, which
    the stand-in lacks, so the names text_encoder takes from transformers
    import first, and the stand-in leaves sys.modules after."""
    from transformers import (AutoModelForCausalLM, AutoProcessor,  # noqa: F401
                              AutoTokenizer, Mistral3ForConditionalGeneration,
                              pipeline)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    if "torchvision" in sys.modules:
        raise SystemExit("error: torchvision is imported")
    sys.modules["torchvision"] = types.ModuleType("torchvision")
    try:
        from flux2 import text_encoder
    finally:
        del sys.modules["torchvision"]
    return text_encoder


def load_encoder(torch, encoder: Path, folder: Path):
    """(model, codec census): Qwen3ForCausalLM in float32 on the CPU with
    the GGUF's weights and no output matrix."""
    import gguf
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, Qwen3ForCausalLM

    cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    reader = gguf.GGUFReader(str(encoder))
    kv = {f.name: f for f in reader.fields.values()}
    for k, key in WIDTHS.items():
        f = kv[f"qwen3.{key}"]
        got = f.parts[f.data[0]].tolist()[0]
        if got != getattr(cfg, k):
            raise SystemExit(f"error: {k} is {got} in the GGUF, "
                             f"{getattr(cfg, k)} in the config")
    with init_empty_weights():
        te = Qwen3ForCausalLM(cfg)
    te.lm_head = torch.nn.Identity()
    lm = te.model
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
    if set(left) - {HEAD} or seen != set(want):
        raise SystemExit(f"error: GGUF tensors left over {left[:4]}, model "
                         f"parameters not set {sorted(set(want) - seen)[:4]}")
    meta = [n for n, p in te.named_parameters() if p.is_meta]
    kinds = {p.dtype for p in te.parameters()} | {
        b.dtype for b in te.buffers() if b.is_floating_point()}
    if meta or kinds != {torch.float32}:
        raise SystemExit(f"error: parameters on no device {meta[:4]}, "
                         f"dtypes {kinds}")
    if len(lm.layers) != 36:
        raise SystemExit(f"error: {len(lm.layers)} layers, expected 36")
    return te.eval(), census


class Keep:
    """The model, keeping the inputs and output of its last call, the
    outputs of the decoder layers ``taps`` and the rotary positions."""

    def __init__(self, model, taps):
        self.model = model
        self.last = None
        self.layers = {}
        self.positions = None
        inner = model.model
        for i in taps:
            inner.layers[i].register_forward_hook(self._layer(i))
        inner.rotary_emb.register_forward_pre_hook(self._rope,
                                                   with_kwargs=True)

    def _layer(self, i):
        def hook(_mod, _args, out):
            self.layers[i] = out[0] if isinstance(out, tuple) else out
        return hook

    def _rope(self, _mod, args, kwargs):
        pos = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        self.positions = pos

    @property
    def device(self):
        return self.model.device

    def __call__(self, **kwargs):
        self.layers, self.positions = {}, None
        out = self.model(**kwargs)
        self.last = (kwargs, out)
        return out


def embedder(torch, text_encoder, keep, tok):
    """An instance of the reference's Qwen3Embedder on ``keep`` and
    ``tok``: its __init__ loads from a hub id, so it is set by hand as
    __init__ sets it."""
    enc = object.__new__(text_encoder.Qwen3Embedder)
    torch.nn.Module.__init__(enc)
    enc.model = keep
    enc.tokenizer = tok
    enc.max_length = text_encoder.MAX_LENGTH
    return enc


def features(torch, text_encoder, enc, prompt: str) -> dict:
    t0 = time.perf_counter()
    out = enc([prompt])
    secs = time.perf_counter() - t0
    keep = enc.model
    kw, res = keep.last
    hs = res.hidden_states
    layers = text_encoder.OUTPUT_LAYERS_QWEN3
    tapped = {k: bool(torch.equal(hs[k], keep.layers[k - 1]))
              for k in layers}
    if not all(tapped.values()):
        raise SystemExit(f"error: hidden_states and the layer outputs "
                         f"differ: {tapped}")
    length = kw["input_ids"].shape[1]
    f = torch.stack([hs[k] for k in layers], dim=1)
    f = f.permute(0, 2, 1, 3).reshape(length, -1)
    if not torch.equal(f, out[0]):
        raise SystemExit("error: the stack differs from forward's own")
    mask = kw["attention_mask"][0]
    real = int(mask.sum())
    f32 = oracle.to_numpy(out[0])
    pads = f32[real:].astype(np.float64)
    spread = [0.0, 0.0]
    if len(pads) > 1:
        r = (np.linalg.norm(pads[1:] - pads[0], axis=1)
             / np.linalg.norm(pads[0]))
        spread = [float(r.min()), float(r.max())]
    pos = keep.positions
    pos = (np.arange(length)[None] if pos is None
           else pos.numpy().astype(np.int64))
    return {"ids": kw["input_ids"][0].numpy().astype(np.int64),
            "mask": mask.numpy().astype(np.int64),
            "positions": pos.astype(np.int64),
            "positions_given": keep.positions is not None,
            "features": f32, "real": real, "pad_spread": spread,
            "seconds": secs}


def rendered(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def one_pad(tok, cap: int) -> str:
    """The longest prefix of the long prompt, cut at a character, whose
    templated prompt is one token under the cap, so its features have one
    pad row."""
    for n in range(len(_LONG), 0, -1):
        p = _LONG[:n].rstrip()
        if len(tok(rendered(tok, p))["input_ids"]) == cap - 1:
            return p
    raise SystemExit(f"error: no prefix of the long prompt is {cap - 1} "
                     "tokens")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, type=Path,
                    help="src directory of BFL's flux2")
    ap.add_argument("--reference-commit", required=True,
                    help="commit of --reference, for the manifest")
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-4B text GGUF")
    ap.add_argument("--text-encoder", required=True, type=Path,
                    help="the official repo's text_encoder folder")
    ap.add_argument("--tokenizer", required=True, type=Path,
                    help="the official repo's tokenizer folder")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    text_encoder = reference(a.reference.expanduser())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(a.tokenizer.expanduser()),
                                        local_files_only=True)
    cap = text_encoder.MAX_LENGTH
    prompts = dict(PROMPTS)
    prompts["one_pad"] = one_pad(tok, cap)
    t0 = time.perf_counter()
    te, census = load_encoder(torch, a.encoder, a.text_encoder.expanduser())
    load_s = time.perf_counter() - t0
    print(f"[check2] loaded {a.encoder.name} in {load_s:.1f}s, codecs "
          f"{census}", flush=True)
    fx = fixtures.FixtureWriter(a.name, check=2, family="flux2_klein",
                                script=SCRIPT)
    fx.add_file("encoder", a.encoder)
    fx.add_file("text_encoder/config.json", a.text_encoder / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja"):
        fx.add_file(f"tokenizer/{f}", a.tokenizer / f)
    layers = list(text_encoder.OUTPUT_LAYERS_QWEN3)
    fx.inputs.update(prompts=prompts,
                     rendered={k: rendered(tok, p)
                               for k, p in prompts.items()},
                     tap_layers=layers, max_length=cap,
                     reference_commit=a.reference_commit,
                     gguf_codecs=census,
                     matmul_bf16=("bf16 operands, float32 GEMM, one bf16 "
                                  "rounding"))
    enc = embedder(torch, text_encoder, Keep(te, [k - 1 for k in layers]),
                   tok)
    runs = {}
    for mode in ("fp32", "bf16"):
        if mode == "bf16":
            # A model loaded in bf16 keeps its rotary frequencies in fp32,
            # which a cast of the fp32 model would round.
            rot = te.model.rotary_emb
            inv = rot.inv_freq.clone()
            te.to(torch.bfloat16)
            rot.inv_freq = inv
            oracle.emulate_bf16_matmul(True)
        runs[mode] = {}
        for k, v in prompts.items():
            runs[mode][k] = features(torch, text_encoder, enc, v)
            print(f"[check2] {mode} {k}: "
                  f"{runs[mode][k]['seconds']:.1f}s", flush=True)
    oracle.emulate_bf16_matmul(False)
    tol = {"rel_l2": {}, "rel_l2_real": {}, "rel_l2_pads": {},
           "token_cosine_min": {}}
    rows, secs, pads, given = {}, {}, {}, {}
    for k in prompts:
        f32, b16 = runs["fp32"][k], runs["bf16"][k]
        for key in ("ids", "mask", "positions"):
            assert np.array_equal(f32[key], b16[key]), (k, key)
            fx.add(f"{key}_{k}", f32[key])
        fx.add(f"features_{k}", f32["features"])
        fx.add(f"features_{k}_bf16", b16["features"])
        n = f32["real"]
        got, ref = b16["features"], f32["features"]
        gap = fixtures.rel_l2(got, ref)
        real = fixtures.rel_l2(got[:n], ref[:n])
        c = token_cosine(got, ref)
        tol["rel_l2"][k] = 2 * gap
        tol["rel_l2_real"][k] = 2 * real
        tol["token_cosine_min"][k] = float(1 - 4 * (1 - c[:n].min()))
        res = {"rel_l2": gap, "rel_l2_real": real,
               "token_cosine_min_real": float(c[:n].min()),
               "worst_real_token": int(c[:n].argmin())}
        if n < len(c):
            res.update(rel_l2_pads=fixtures.rel_l2(got[n:], ref[n:]),
                       token_cosine_min_pads=float(c[n:].min()),
                       worst_pad_row=n + int(c[n:].argmin()))
            tol["rel_l2_pads"][k] = 2 * res["rel_l2_pads"]
        fx.results[f"bf16_vs_fp32_{k}"] = res
        rows[k] = {"real": n, "rows": int(f32["ids"].shape[0])}
        pads[k] = f32["pad_spread"]
        given[k] = f32["positions_given"]
        secs[k] = {m: round(runs[m][k]["seconds"], 1) for m in runs}
        print(f"[check2] {k}: {rows[k]}, pad rows {pads[k][0]:.1e} to "
              f"{pads[k][1]:.1e} from the first, bf16 vs fp32 rel L2 "
              f"{gap:.3e}, real rows {real:.3e}, pad rows "
              f"{res.get('rel_l2_pads', 0.0):.3e}, min token cosine real "
              f"{c[:n].min():.5f}, pads "
              f"{res.get('token_cosine_min_pads', 1.0):.5f}", flush=True)
    if rows["long"]["real"] != cap or rows["one_pad"]["real"] != cap - 1:
        raise SystemExit(f"error: rows {rows}")
    fx.inputs["rows"] = rows
    fx.results["pad_rows_rel_l2_to_the_first"] = pads
    fx.results["positions_from_the_model"] = given
    fx.results["seconds"] = {"load_s": round(load_s, 1), **secs}
    fx.results["oracle"] = {"device": "cpu",
                            "threads": torch.get_num_threads()}
    fx.tolerances.update(**tol, rule=RULE)
    fx.note("positions_from_the_model is whether the rotary embedding "
            "took position ids from the model's forward; the positions "
            "recorded are those ids, else the row indices.")
    print(f"[check2] wrote {fx.write()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
