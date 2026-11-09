#!/usr/bin/env python3
"""Goldens of a tiny random EmbeddingGemma 2, committed for the CPU test
tests/models/test_gemma_embedding2.py.

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

``media.npz`` holds the same backbone with tiny random Gemma 4 vision and
audio towers (``config`` again, and ``w.<name>`` for the tower weights
under transformers' names). The model runs on the inputs the reference
processor would hand it: patches padded to the patch budget with positions
of -1, every frame of a video in one batch, and every clip of an item in
one padded batch. Per case it stores ``<case>.ids``, the inputs, ``.vec``
and ``.gap``:

- ``image``: two images of 48x96 and 96x48 (2 soft tokens each), padded
  to 27 patches; ``<case>.image<i>`` is the uint8 image;
- ``video``: three 96x96 frames (4 soft tokens each), padded to 45
  patches; ``video.frames``;
- ``audio``: clips of 1.0 s and 0.61 s in one padded batch;
  ``audio.wave<i>`` is the int16 clip, ``audio.features<i>`` and
  ``audio.mask<i>`` the reference extractor's output for it alone;
- ``mixed``: text, an image, a clip and a video in one item.

transformers' audio attention has no float64 run, so the cases with audio
store a gap of 0.
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


# The media token ids, inside the tiny vocabulary.
MEDIA_IDS = dict(image_token_id=90, video_token_id=91, audio_token_id=92,
                 boi_token_id=93, eoi_token_id=94, boa_token_id=95,
                 eoa_token_index=89)

VISION = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=2, head_dim=16,
              global_head_dim=16, patch_size=16, pooling_kernel_size=3,
              position_embedding_size=64, default_output_length=4,
              rms_norm_eps=1e-6, standardize=False, use_clipped_linears=False)

AUDIO = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
             subsampling_conv_channels=[128, 4], conv_kernel_size=5,
             attention_chunk_size=12, attention_context_left=13,
             attention_context_right=0, output_proj_dims=24,
             rms_norm_eps=1e-6, use_clipped_linears=True)

PATCH = 16


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


def build_media(torch, text_model):
    from transformers.models.embedding_gemma2 import (
        configuration_embedding_gemma2 as cfg_mod,
        modeling_embedding_gemma2 as mod,
    )
    from transformers.models.gemma4 import configuration_gemma4 as g4

    cfg = cfg_mod.EmbeddingGemma2Config(
        text_config=text_model.config.to_dict(),
        vision_config=g4.Gemma4VisionConfig(**VISION).to_dict(),
        audio_config=g4.Gemma4AudioConfig(**AUDIO).to_dict(), **MEDIA_IDS)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg.audio_config._attn_implementation = "eager"
    torch.manual_seed(SEED + 1)
    model = mod.EmbeddingGemma2Model(cfg).eval()
    model.language_model.load_state_dict(text_model.state_dict())
    gen = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("language_model."):
                continue
            if "norm" in name or name.endswith("per_dim_scale"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif p.ndim == 1:
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            else:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=gen) / fan_in ** 0.5)
        # Finite clip bounds, so the audio clamps act.
        for name, b in model.named_buffers():
            if name.endswith(("input_min", "output_min")):
                b.copy_(torch.tensor(-2.0 - torch.rand((), generator=gen).item()))
            elif name.endswith(("input_max", "output_max")):
                b.copy_(torch.tensor(2.0 + torch.rand((), generator=gen).item()))
    return cfg, model


def patches(img: np.ndarray, budget: int):
    """transformers' Gemma 4 patchify of an [H, W, 3] uint8 image, rescaled,
    padded to ``budget`` patches with positions of -1."""
    x = img.astype(np.float64).transpose(2, 0, 1) / 255.0
    c, h, w = x.shape
    ph, pw = h // PATCH, w // PATCH
    p = x.reshape(c, ph, PATCH, pw, PATCH).transpose(1, 3, 2, 4, 0)
    p = p.reshape(ph * pw, -1)
    gx, gy = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    pos = np.stack([gx, gy], -1).reshape(-1, 2)
    pad = budget - p.shape[0]
    p = np.concatenate([p, np.zeros((pad, p.shape[1]))])
    pos = np.concatenate([pos, np.full((pad, 2), -1)])
    return p, pos, ph * pw // 9


def synthetic_clip(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 16000.0
    x = 0.05 * rng.standard_normal(samples)
    for i in range(4):
        x += 0.2 * np.sin(2 * np.pi * (220.0 + 170.0 * i) * t + i)
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def audio_tokens(mask: np.ndarray) -> int:
    """The processor's count: the feature mask after two stride-2 steps."""
    m = mask
    for _ in range(2):
        t = len(m)
        m = m[::2][:(t + 2 - 3) // 2 + 1]
    return int(m.sum())


def record_media(torch, cfg, model, out):
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )

    ids = MEDIA_IDS
    rng = np.random.default_rng(SEED + 2)

    def img(h, w):
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def vision_block(tok, n):
        return [ids["boi_token_id"]] + [tok] * n + [ids["eoi_token_id"]]

    def audio_block(n):
        return [ids["boa_token_id"]] + [ids["audio_token_id"]] * n + [ids["eoa_token_index"]]

    def text(n):
        return list(rng.integers(3, 89, n))

    fe = Gemma4AudioFeatureExtractor()
    images = [img(48, 96), img(96, 48)]
    frames = [img(96, 96) for _ in range(3)]
    waves = [synthetic_clip(16000, 1), synthetic_clip(9760, 2)]

    def image_inputs(imgs, budget=27):
        ps = [patches(i, budget) for i in imgs]
        return (np.stack([p for p, _, _ in ps]), np.stack([q for _, q, _ in ps]),
                [n for _, _, n in ps])

    def audio_inputs(ws):
        f = fe([w.astype(np.float32) / 32768.0 for w in ws], return_tensors="np")
        return f["input_features"], f["input_features_mask"]

    pv, ppos, n_img = image_inputs(images)
    vv, vpos, n_frame = image_inputs(frames, budget=45)
    af, am = audio_inputs(waves)
    n_aud = [audio_tokens(am[i]) for i in range(len(waves))]
    for i, w in enumerate(waves):
        f1, m1 = audio_inputs([w])
        out[f"audio.features{i}"] = f1[0].astype(np.float32)
        out[f"audio.mask{i}"] = m1[0].astype(bool)
    for i, im in enumerate(images):
        out[f"image.image{i}"] = im
    out["video.frames"] = np.stack(frames)
    for i, w in enumerate(waves):
        out[f"audio.wave{i}"] = w

    bos, eos = [2], [1]
    plans = {
        "image": (bos + text(3) + vision_block(ids["image_token_id"], n_img[0])
                  + text(2) + vision_block(ids["image_token_id"], n_img[1]) + eos,
                  dict(pixel_values=pv, image_position_ids=ppos)),
        "video": (bos + text(2) + sum((vision_block(ids["video_token_id"], n)
                                       for n in n_frame), []) + eos,
                  dict(pixel_values_videos=vv, video_position_ids=vpos,
                       num_frames_per_video=np.array([3]))),
        "audio": (bos + audio_block(n_aud[0]) + text(4) + audio_block(n_aud[1]) + eos,
                  dict(input_features=af, input_features_mask=am)),
    }
    mpv, mpos, _ = image_inputs(images[:1])
    maf, mam = audio_inputs(waves[1:])
    plans["mixed"] = (
        bos + text(2) + vision_block(ids["image_token_id"], n_img[0]) + text(3)
        + audio_block(n_aud[1]) + text(1)
        + sum((vision_block(ids["video_token_id"], n) for n in n_frame), []) + eos,
        dict(pixel_values=mpv, image_position_ids=mpos, input_features=maf,
             input_features_mask=mam, pixel_values_videos=vv,
             video_position_ids=vpos, num_frames_per_video=np.array([3])))

    def run(dtype, ids_, kw):
        m = model.to(dtype)
        t = {}
        for k, v in kw.items():
            if v.dtype == np.bool_ or "position" in k or k.startswith("num_"):
                t[k] = torch.tensor(v)
            else:
                t[k] = torch.tensor(v, dtype=dtype)
        with torch.no_grad():
            o = m(input_ids=torch.tensor([ids_]), **t)
        v = o.last_hidden_state[0].double().mean(0)
        return (v / v.norm()).numpy()

    for case, (ids_, kw) in plans.items():
        v32 = run(torch.float32, ids_, kw)
        try:
            gap = rel_l2(v32, run(torch.float64, ids_, kw))
        except RuntimeError:
            # transformers' audio attention builds a float32 tensor, so
            # the tower has no float64 run; the test takes its floor.
            gap = 0.0
        model.to(torch.float32)
        out[f"{case}.ids"] = np.array(ids_, dtype=np.int64)
        out[f"{case}.vec"] = v32.astype(np.float32)
        out[f"{case}.gap"] = np.array([gap])
        print(case, len(ids_), "tokens, gap", out[f"{case}.gap"])


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

    _, model = build(torch, tf)
    cfg, media = build_media(torch, model)
    out = {"config": np.array(json.dumps(dict(
        MEDIA_IDS, text_config=dict(CONFIG, layer_types=cfg.text_config.layer_types),
        vision_config=VISION, audio_config=AUDIO)))}
    for name, p in media.state_dict().items():
        if name.startswith("language_model.") or "rotary_emb" in name:
            continue
        out[f"w.{name}"] = p.float().numpy()
    record_media(torch, cfg, media, out)
    np.savez_compressed(args.out / "media.npz", **out)


if __name__ == "__main__":
    main()
