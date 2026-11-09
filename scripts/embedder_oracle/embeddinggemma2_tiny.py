#!/usr/bin/env python3
"""Goldens of a tiny random EmbeddingGemma 2 text backbone, committed for
the CPU test tests/models/test_gemma_embedding2.py.

  <oracle python> -I scripts/embedder_oracle/embeddinggemma2_tiny.py \\
      --out tests/fixtures/embeddinggemma2_oracle

The oracle environment holds torch, transformers 5.22.2 and numpy, and
never gmlx or mlx. The script seeds a tiny ``EmbeddingGemma2TextModel``
(sliding radius 8, full layers at 3 and 6), runs it in float32 and
float64, and writes ``text.npz``:

- ``config``: the text config as JSON;
- ``w.<name>``: every weight, under gmlx's ``language_model.`` names;
- per case, ``<case>.ids`` or ``<case>.embeds``, the float32 normed last
  hidden states ``<case>.hidden``, and the sentence vector ``<case>.vec``
  (sentence-transformers' mean pooling and normalization);
- ``<case>.gap``: the float32 run's relative L2 distance from float64.

Cases: ``short`` (7 tokens, under one window), ``long`` (45 tokens, past
one 16-row block plus twice the window), and ``soft`` (45 tokens with 12
rows replaced by random soft tokens, as a media merge does).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRANSFORMERS = "5.22.2"
SEED = 7

CONFIG = dict(
    vocab_size=96, hidden_size=64, intermediate_size=96,
    num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
    sliding_window_pattern=3, sliding_window=8,
    hidden_size_per_layer_input=16, embedding_dim=48,
    max_position_embeddings=256, rms_norm_eps=1e-6,
    pad_token_id=0, eos_token_id=1, bos_token_id=2,
)


def build(torch, tf):
    from transformers.models.embedding_gemma2 import (
        configuration_embedding_gemma2 as cfg_mod,
        modeling_embedding_gemma2 as mod,
    )

    cfg = cfg_mod.EmbeddingGemma2TextConfig(**CONFIG)
    cfg._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = mod.EmbeddingGemma2TextModel(cfg).eval()
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layer_scalar"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            elif "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif name == "embed_tokens.weight":
                p.copy_(torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[-1] ** 0.5)
    return cfg, model


def cases():
    rng = np.random.default_rng(SEED)

    def ids(n):
        return np.concatenate([[2], rng.integers(3, CONFIG["vocab_size"], n - 2), [1]])

    long_ids = ids(45)
    soft_rows = np.sort(rng.choice(np.arange(1, 44), 12, replace=False))
    soft = rng.standard_normal((len(soft_rows), CONFIG["hidden_size"]))
    return {"short": ids(7), "long": long_ids}, (long_ids, soft_rows, soft)


def run(torch, model, dtype, ids=None, embeds=None):
    model = model.to(dtype)
    captured = {}
    hook = model.norm.register_forward_hook(
        lambda m, a, out: captured.__setitem__("hidden", out))
    with torch.no_grad():
        if embeds is not None:
            out = model(inputs_embeds=torch.tensor(embeds[None], dtype=dtype))
        else:
            out = model(input_ids=torch.tensor(ids[None]))
    hook.remove()
    # sentence-transformers: mean over every token, then normalize.
    proj = out.last_hidden_state[0].double()
    vec = proj.mean(0)
    vec = vec / vec.norm()
    return captured["hidden"][0].double().numpy(), vec.numpy()


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    import transformers as tf

    if tf.__version__ != TRANSFORMERS:
        raise SystemExit(f"transformers {tf.__version__}, want {TRANSFORMERS}")
    torch.set_num_threads(1)
    cfg, model = build(torch, tf)
    out = {"config": np.array(json.dumps(dict(CONFIG, layer_types=cfg.layer_types)))}
    for name, p in model.state_dict().items():
        if "rotary_emb" in name:
            continue
        out[f"w.language_model.{name}"] = p.float().numpy()

    plain, (long_ids, soft_rows, soft) = cases()
    for case, ids in plain.items():
        h32, v32 = run(torch, model, torch.float32, ids=ids)
        h64, v64 = run(torch, model, torch.float64, ids=ids)
        out[f"{case}.ids"] = ids.astype(np.int64)
        out[f"{case}.hidden"] = h32.astype(np.float32)
        out[f"{case}.vec"] = v32.astype(np.float32)
        out[f"{case}.gap"] = np.array([rel_l2(h32, h64), rel_l2(v32, v64)])

    # Soft tokens: the scaled token embeddings with some rows replaced.
    with torch.no_grad():
        model = model.to(torch.float64)
        embeds = model.embed_tokens(torch.tensor(long_ids[None]))[0].numpy().copy()
    embeds[soft_rows] = soft
    h32, v32 = run(torch, model, torch.float32, embeds=embeds)
    h64, v64 = run(torch, model, torch.float64, embeds=embeds)
    out["soft.embeds"] = embeds.astype(np.float32)
    out["soft.hidden"] = h32.astype(np.float32)
    out["soft.vec"] = v32.astype(np.float32)
    out["soft.gap"] = np.array([rel_l2(h32, h64), rel_l2(v32, v64)])

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "text.npz", **out)
    for case in ("short", "long", "soft"):
        print(case, "gap", out[f"{case}.gap"])


if __name__ == "__main__":
    main()
