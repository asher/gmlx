# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A tiny LFM3 GGUF for header tests: the family's metadata keys and the
three tensors the config synthesis reads shapes from. It holds no model
weights and no template text. Also the lookup of a real LFM3 model and its
projector, for the integration tier, and the float16 weights an M1 or M2
loads."""

from __future__ import annotations

import numpy as np

ARCH = "lfm3"
KV_HEADS = [0, 0, 2, 0, 2]
SLIDING = [False, False, False, False, True]
# The layer schedule and attention geometry of the shipped 2B models:
# conv at 0, 1 and every odd layer, sliding attention at 4, 8, ..., 20,
# full attention at 2, 6, ..., 22.
SHIPPED_KV_HEADS = [0, 0] + [0 if i % 2 else 4 for i in range(2, 24)]
SHIPPED_SLIDING = [i >= 4 and i % 4 == 0 for i in range(24)]


def write_lfm3_header(path, *, context: int = 128000, templates=(),
                      causal: bool | None = None,
                      decision: str | None = None,
                      shipped: bool = False) -> str:
    """Write the GGUF and return its path. ``templates`` are the names in
    ``tokenizer.chat_templates``. ``causal`` and ``decision`` are the two
    keys a d1 decision model carries. ``shipped`` writes the geometry of
    the shipped 2B models (24 layers, 16 query and 4 KV heads, head dim
    128, window 256) in place of the tiny one."""
    import gguf

    kv_heads = SHIPPED_KV_HEADS if shipped else KV_HEADS
    sliding = SHIPPED_SLIDING if shipped else SLIDING
    hidden, heads, head_dim, ffn, window = (
        (2048, 16, 128, 7680, 256) if shipped else (64, 4, 16, 128, 3))
    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_uint32(f"{ARCH}.embedding_length", hidden)
    w.add_uint32(f"{ARCH}.block_count", len(kv_heads))
    w.add_uint32(f"{ARCH}.attention.head_count", heads)
    w.add_array(f"{ARCH}.attention.head_count_kv", list(kv_heads))
    w.add_uint32(f"{ARCH}.feed_forward_length", ffn)
    w.add_uint32(f"{ARCH}.context_length", context)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32(f"{ARCH}.attention.key_length", head_dim)
    w.add_uint32(f"{ARCH}.attention.value_length", head_dim)
    w.add_uint32(f"{ARCH}.attention.sliding_window", window)
    w.add_array(f"{ARCH}.attention.sliding_window_pattern", list(sliding))
    w.add_float32(f"{ARCH}.rope.freq_base", 1000000.0)
    w.add_float32(f"{ARCH}.rope.freq_base_swa", 10000.0)
    w.add_uint32(f"{ARCH}.shortconv.l_cache", 7)
    w.add_array("tokenizer.ggml.tokens", ["t"] * 32)
    if templates:
        w.add_array("tokenizer.chat_templates", list(templates))
    if causal is not None:
        w.add_bool(f"{ARCH}.attention.causal", causal)
    if decision is not None:
        w.add_string(f"{ARCH}.decision.type", decision)
    half = head_dim // 2
    w.add_tensor("rope_freqs.weight",
                 np.array([1.0] * (half // 2) + [1e30] * (half - half // 2),
                          dtype=np.float32))
    for name in ("post_attention_norm", "post_ffw_norm"):
        w.add_tensor(f"blk.0.{name}.weight",
                     np.ones(hidden, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def find_vision_pair(gguf_dir, gguf_index) -> tuple[str, str] | None:
    """The (text GGUF, projector) of a real LFM3 model with vision under
    the integration folder, matched by the base name both headers carry.
    The projector can sit in any subfolder."""
    from gmlx.load.headerscan import scan_gguf

    texts: dict = {}
    for path in gguf_index.get(ARCH, ()):
        kv = scan_gguf(path, include_tensors=False).kv
        texts.setdefault(kv.get("general.basename"), path)
    for path in sorted(gguf_dir.rglob("*mmproj*.gguf")):
        kv = scan_gguf(str(path), include_tensors=False).kv
        if (kv.get("clip.vision.projector_type") == "lfm3vl"
                and kv.get("general.basename") in texts):
            return texts[kv["general.basename"]], str(path)
    return None


def stream_distance(got, want) -> tuple[float, float]:
    """Relative RMS error and one minus the mean cosine over the feature
    rows."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    cosine = (got * want).sum(-1) / (
        np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return (float(np.linalg.norm(got - want) / np.linalg.norm(want)),
            float(1.0 - cosine.mean()))


def greedy_tokens(model, input_ids, steps: int,
                  **inputs) -> tuple[list[int], list[bool]]:
    """The first ``steps`` greedy tokens after a prompt with media, and
    for each step whether its two best logits were a near tie: at most
    two steps of the 16-bit logits apart."""
    import mlx.core as mx

    ids = mx.array(np.asarray(input_ids).reshape(1, -1))
    pixel_values = inputs.pop("pixel_values", None)
    embeds = model.get_input_embeddings(
        ids, pixel_values, **inputs).inputs_embeds
    cache = model.make_cache()
    logits = model.language_model(ids, inputs_embeds=embeds,
                                  cache=cache).logits
    tokens, ties = [], []
    for _ in range(steps):
        row = logits[0, -1].astype(mx.float32)
        second, best = (float(v) for v in mx.sort(row)[-2:].tolist())
        # bfloat16 keeps 8 significant bits.
        step = 2.0 ** (np.floor(np.log2(max(abs(best), 1e-6))) - 7)
        ties.append(best - second <= 2 * step)
        token = mx.argmax(row)
        tokens.append(int(token.item()))
        logits = model.language_model(token.reshape(1, 1),
                                      cache=cache).logits
    return tokens, ties


def same_answer_up_to_a_tie(a, b) -> bool:
    """Whether two ``greedy_tokens`` results agree, or first differ at a
    step that one of them holds as a near tie. After such a step the two
    are different continuations."""
    for i, (x, y) in enumerate(zip(a[0], b[0])):
        if x != y:
            return a[1][i] or b[1][i]
    return True


def load_as_f16_projector(text_path, projector_path):
    """Load the pair as an M1 or M2 does with the F16 projector: call
    under ``GMLX_ACTIVATION_DTYPE=float16``. The activation cast never
    visits BF16 tensors, so a BF16 projector's matrices are cast here.
    Without that the mix of float16 and bfloat16 runs a float32 stream,
    which cannot overflow."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from gmlx.load.vlm import load_vlm_model

    model, config, processor = load_vlm_model(text_path, projector_path)
    for name in ("vision_tower", "multi_modal_projector", "audio_encoder",
                 "audio_projector"):
        module = getattr(model, name, None)
        if module is not None:
            module.update(tree_unflatten([
                (k, v.astype(mx.float32).astype(mx.float16))
                for k, v in tree_flatten(module.parameters())
                if v.dtype == mx.bfloat16]))
    return model, config, processor


def write_projector_header(path, *, arch: str = "clip", vision: bool = False,
                           audio: bool = False, speech: bool = False) -> str:
    """Write a projector GGUF that holds only the keys saying what it
    gives its text model, and return its path."""
    import gguf

    w = gguf.GGUFWriter(str(path), arch)
    for key, on in (("clip.has_vision_encoder", vision),
                    ("clip.has_audio_encoder", audio),
                    ("clip.has_gen_audio_encoder", speech)):
        if on:
            w.add_bool(key, True)
    w.add_tensor("a.weight", np.zeros(1, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)
