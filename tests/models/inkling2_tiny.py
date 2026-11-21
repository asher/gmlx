"""Tiny Inkling 2 GGUFs in both layouts, written from one set of values.

The looped layout ('inkling2') writes one block per layer visit, packs the
loop message queries and stores the controllers as BF16. The mainline
layout ('inkling2_loop') writes the physical layers, one message query
tensor per core layer and F32 controllers. The controller values are
rounded to BF16 first, so both files hold the same numbers.
"""
from __future__ import annotations

import numpy as np

D, HEADS, KV, HD, R = 64, 4, 2, 16, 4
LAYERS, START, END, PASSES, BLOCK, RANK = 8, 2, 6, 3, 2, 8
EXPERTS, USED, FF = 8, 2, 32
WINDOW, EXTENT, TAPS = 16, 16, 4
SWA = [True, False] * 4


def visits() -> list[int]:
    return (list(range(END)) + list(range(START, END)) * (PASSES - 1)
            + list(range(END, LAYERS)))


def tokens() -> list[str]:
    from tokenizers import pre_tokenizers
    return (["<|end|>", "<|pad|>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
            + ["He", "wo"])


def _bf16(a: np.ndarray) -> np.ndarray:
    """``a`` rounded to the nearest BF16 value, as float32."""
    bits = a.astype(np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def values(seed: int = 0, scales=None) -> dict[str, np.ndarray]:
    """Physical-layer tensors by GGUF name, in numpy (row-major) order.
    ``scales`` (layer -> per-expert gate_up scale) adds the
    ``ffn_gate_up_exps.scale`` tensors."""
    rng = np.random.default_rng(seed)
    n_vocab = len(tokens())
    padded = n_vocab + 6

    def t(*shape, scale=0.1, base=0.0):
        return (base + rng.standard_normal(shape) * scale).astype(np.float32)

    v = {
        "token_embd.weight": t(padded, D, scale=0.5),
        "token_embd_norm.weight": t(D, base=1.0),
        "output_norm.weight": t(D, base=1.0),
        "output_res_query.weight": t(D),
        "output.weight": t(padded, D),
    }
    core = END - START
    v["loop_res_query.weight"] = t(2 * core, D)
    v["loop_kv_res_query.weight"] = t(2 * core, D)
    for i in range(LAYERS):
        p = f"blk.{i}."
        full = not SWA[i]
        if full:
            for n in ("q", "k", "v"):
                v[p + f"attn_{n}_inp_norm.weight"] = t(D, base=1.0)
            v[p + "attn_k_conv_proj.weight"] = t(KV * TAPS, D)
            v[p + "attn_v_conv_proj.weight"] = t(KV * TAPS, D)
        else:
            v[p + "attn_norm.weight"] = t(D, base=1.0)
        if i:
            v[p + "attn_res_query.weight"] = t(3, D) if full else t(D)
        v[p + "ffn_res_query.weight"] = t(D)
        v[p + "attn_q.weight"] = t(HEADS * HD, D)
        v[p + "attn_k.weight"] = t(KV * HD, D)
        v[p + "attn_v.weight"] = t(KV * HD, D)
        v[p + "attn_gate.weight"] = t(HEADS * HD, D)
        v[p + "attn_output.weight"] = t(D, HEADS * HD)
        v[p + "attn_rel.weight"] = t(HEADS * R, D)
        v[p + "attn_sink_proj.weight"] = t(HEADS, D)
        v[p + "attn_q_clip.weight"] = t(HEADS, HD, base=1.0)
        v[p + "attn_rel_clip.weight"] = t(HEADS, R, base=1.0)
        v[p + "attn_rel_bias.weight"] = t(EXTENT, R, scale=0.5)
        v[p + "attn_k_conv.weight"] = t(KV * HD, TAPS, scale=0.3)
        v[p + "attn_v_conv.weight"] = t(KV * HD, TAPS, scale=0.3)
        v[p + "attn_post_conv.weight"] = t(D, TAPS, scale=0.3)
        v[p + "ffn_post_conv.weight"] = t(D, TAPS, scale=0.3)
        v[p + "ffn_norm.weight"] = t(D, base=1.0)
        v[p + "ffn_gate_inp.weight"] = t(EXPERTS, D, scale=0.5)
        v[p + "exp_probs_b.bias"] = t(EXPERTS)
        v[p + "ffn_gate_up_exps.weight"] = t(EXPERTS, 2 * FF, D, scale=0.3)
        v[p + "ffn_down_exps.weight"] = t(EXPERTS, D, FF, scale=0.3)
        if scales and i in scales:
            v[p + "ffn_gate_up_exps.scale"] = np.asarray(scales[i], np.float32)
        if START <= i < END:
            for c in ("q", "k", "v", "ffn"):
                v[p + f"loop.{c}.down"] = _bf16(t(RANK, 2 * D))
                v[p + f"loop.{c}.gain"] = _bf16(t(D, RANK))
                v[p + f"loop.{c}.forget"] = _bf16(t(D, RANK))
    return v


_ROOT_CTRL = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}


def _tensor_names(v: dict, layout: str):
    """(GGUF name, array, bf16) for every tensor of ``layout``."""
    core = END - START
    q = v["loop_res_query.weight"]
    kv = v["loop_kv_res_query.weight"]
    for name, a in v.items():
        if name.startswith("loop_"):
            continue
        if ".loop." in name:
            blk, rest = name.split(".loop.")
            c, part = rest.split(".")
            if layout == "root":
                yield f"{blk}.loop_{_ROOT_CTRL[c]}_{part}.weight", a, True
            else:
                yield f"{blk}.loop_{c}_ctrl_{part}.weight", a, False
            continue
        yield name, a, False
    if layout == "root":
        yield "loop_res_query.weight", q, False
        yield "loop_kv_res_query.weight", kv, False
    else:
        for j in range(core):
            msg = np.stack([q[2 * j], kv[2 * j], kv[2 * j + 1], q[2 * j + 1]])
            yield f"blk.{START + j}.loop_msg_query.weight", msg, False


def metadata(layout: str, padded: int) -> dict:
    """The GGUF header of ``layout`` ('root' or 'mainline') as plain
    values, for ``synthesize_config`` or ``write``."""
    arch = "inkling2" if layout == "root" else "inkling2_loop"
    order = visits()
    per = (lambda vals: [vals[p] for p in order]) if layout == "root" else list
    toks = tokens()
    m = {"general.architecture": arch}
    fields = {
        "block_count": len(order) if layout == "root" else LAYERS,
        "embedding_length": D,
        "feed_forward_length": FF,
        "context_length": 4096,
        "attention.head_count": per([HEADS] * LAYERS),
        "attention.head_count_kv": per([KV] * LAYERS),
        "attention.key_length": HD,
        "attention.value_length": HD,
        "attention.layer_norm_rms_epsilon": 1e-6,
        "attention.sliding_window_pattern": per(SWA),
        "attention.sliding_window": WINDOW,
        "attention.relative_length": R,
        "attention.relative_extent": EXTENT,
        "attention.qk_clip": 22.627417,
        "attention.relative_clip": 8.0,
        "attention.log_scale_floor": 8,
        "attention.log_scale_alpha": 0.1,
        "attn_res.block_size": BLOCK,
        "shortconv.l_cache": TAPS,
        "activation.situ_beta": 4.0,
        "activation.situ_linear_beta": 25.0,
        "expert_count": EXPERTS,
        "expert_used_count": USED,
        "expert_feed_forward_length": FF,
        "expert_weights_scale": 4.0,
        "final_logit_softcapping": 30.0,
        "vocab_size": padded,
        "vocab_size_unpadded": len(toks),
        "loop.gate_rank": RANK,
    }
    if layout == "root":
        fields.update({"num_loops": PASSES, "loop.core_start": START,
                       "loop.core_end": END})
    else:
        fields.update({"loop.count": PASSES, "loop.start": START,
                       "loop.end": END, "loop.varying": False})
    m.update({f"{arch}.{k}": v for k, v in fields.items()})
    m.update({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": ["H e", "w o"],
        "tokenizer.ggml.token_type": [3, 3] + [1] * (len(toks) - 2),
        "tokenizer.ggml.eos_token_id": 0,
        "tokenizer.ggml.add_bos_token": False,
    })
    return m


def shapes(layout: str, v: dict) -> dict[str, list[int]]:
    """Tensor shapes by GGUF name, in numpy order."""
    return {n: list(a.shape) for n, a, _ in _tensor_names(v, layout)}


def write(path, layout: str, v: dict, *, experts_q8: bool = True,
          meta: dict | None = None, q8: tuple = ()) -> None:
    """Write ``v`` as a ``layout`` GGUF, with header ``meta`` (default
    ``metadata(layout)``). Tensors whose names end with a suffix in ``q8``
    are written as Q8_0."""
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    if meta is None:
        meta = metadata(layout, v["token_embd.weight"].shape[0])
    w = GGUFWriter(str(path), meta["general.architecture"])
    for key, val in meta.items():
        if key == "general.architecture":
            continue
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, int):
            w.add_uint32(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        else:
            w.add_array(key, val)
    for name, a, bf16 in _tensor_names(v, layout):
        if bf16:
            bits = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            w.add_tensor(name, bits, raw_shape=a.shape,
                         raw_dtype=GGMLQuantizationType.BF16)
        elif (experts_q8 and name.endswith("_exps.weight")) or (
                q8 and name.endswith(q8)):
            q = quantize(a, GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
