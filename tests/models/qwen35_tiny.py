"""Tiny Qwen3.5-family GGUFs: a 'qwen35' file in the split or the fused
tensor layout, and a 'sensenova_loop' file in its looped layout.

The fused layout is llama.cpp's shared-input merge: ``attn_qkv_gate`` is
``attn_qkv`` then ``attn_gate`` along the output rows, ``ssm_beta_alpha``
is ``ssm_beta`` then ``ssm_alpha``. Both layouts hold the same values.

The looped file runs physical layers [START, END) PASSES times. It
writes the physical layers before END under their own block, the layers
after the loop under their visit index, and the NextN head after the
last visit.
"""
from __future__ import annotations

import numpy as np

H, FFN, HEADS, KV, HD = 64, 128, 4, 2, 32
HK, HV, DK, DV, CONV = 2, 4, 32, 32, 4
INNER = HV * DV
QKV = 2 * HK * DK + INNER
INTERVAL = 4
SPECIALS = ["<s>", "</s>", "<pad>"]

# qwen35: 4 layers. sensenova_loop: 6 physical layers, [1, 5) twice.
QWEN35_LAYERS = 4
LAYERS, START, END, PASSES = 6, 1, 5, 2
MHC_ITERATIONS = 20


def tokens() -> list[str]:
    from tokenizers import pre_tokenizers
    return SPECIALS + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"]


def visits() -> list[int]:
    return (list(range(START)) + list(range(START, END)) * PASSES
            + list(range(END, LAYERS)))


def _is_linear(i: int) -> bool:
    return (i + 1) % INTERVAL != 0


def _layer(rng, linear: bool, mhc: bool) -> dict[str, np.ndarray]:
    def t(*shape, scale=0.1, base=0.0):
        return (base + rng.standard_normal(shape) * scale).astype(np.float32)

    v = {
        "attn_norm.weight": t(H, base=1.0),
        "post_attention_norm.weight": t(H, base=1.0),
        "ffn_gate.weight": t(FFN, H),
        "ffn_up.weight": t(FFN, H),
        "ffn_down.weight": t(H, FFN),
    }
    if linear:
        v.update({
            "attn_qkv.weight": t(QKV, H),
            "attn_gate.weight": t(INNER, H),
            "ssm_a": -np.exp(t(HV)),
            "ssm_beta.weight": t(HV, H),
            "ssm_alpha.weight": t(HV, H),
            "ssm_conv1d.weight": t(QKV, CONV, scale=0.5),
            "ssm_dt.bias": t(HV),
            "ssm_norm.weight": t(DV, base=1.0),
            "ssm_out.weight": t(H, INNER),
        })
    else:
        v.update({
            "attn_q.weight": t(HEADS * HD * 2, H),
            "attn_k.weight": t(KV * HD, H),
            "attn_v.weight": t(KV * HD, H),
            "attn_output.weight": t(H, HEADS * HD),
            "attn_q_norm.weight": t(HD, base=1.0),
            "attn_k_norm.weight": t(HD, base=1.0),
        })
    if mhc:
        for r in ("attn", "ffn"):
            v[f"hc_{r}_fn.weight"] = t(8, 2 * H, scale=0.5)
            v[f"hc_{r}_fn.bias"] = t(8, scale=0.5)
    return v


def values(arch: str, seed: int = 0) -> dict[str, np.ndarray]:
    """Tensors by GGUF name in the split layout, physical block indices,
    numpy (row-major) order. The NextN head of a sensenova_loop file is
    block LAYERS."""
    rng = np.random.default_rng(seed)
    n = len(tokens())
    v = {
        "token_embd.weight": (rng.standard_normal((n, H)) * 0.5).astype(np.float32),
        "output_norm.weight": (1 + rng.standard_normal(H) * 0.1).astype(np.float32),
        "output.weight": (rng.standard_normal((n, H)) * 0.1).astype(np.float32),
    }
    loop = arch == "sensenova_loop"
    layers = LAYERS if loop else QWEN35_LAYERS
    for i in range(layers):
        mhc = loop and START <= i < END
        for k, a in _layer(rng, _is_linear(i), mhc).items():
            v[f"blk.{i}.{k}"] = a
    if loop:
        for k, a in _layer(rng, False, False).items():
            v[f"blk.{LAYERS}.{k}"] = a
        p = f"blk.{LAYERS}.nextn."
        v[p + "eh_proj.weight"] = (rng.standard_normal((H, 2 * H)) * 0.1).astype(np.float32)
        for k in ("enorm", "hnorm", "shared_head_norm"):
            v[p + f"{k}.weight"] = (1 + rng.standard_normal(H) * 0.1).astype(np.float32)
    return v


def _file_block(arch: str, b: int) -> int:
    if arch != "sensenova_loop" or b < END:
        return b
    return b + (PASSES - 1) * (END - START)


def _tensors(arch: str, v: dict, fused: bool):
    """(file name, array) in the file's layout."""
    for name, a in v.items():
        if not name.startswith("blk."):
            yield name, a
            continue
        _, b, rest = name.split(".", 2)
        blk = f"blk.{_file_block(arch, int(b))}."
        if fused and rest == "attn_qkv.weight":
            gate = v[f"blk.{b}.attn_gate.weight"]
            yield blk + "attn_qkv_gate.weight", np.concatenate([a, gate])
        elif fused and rest == "ssm_beta.weight":
            alpha = v[f"blk.{b}.ssm_alpha.weight"]
            yield blk + "ssm_beta_alpha.weight", np.concatenate([a, alpha])
        elif fused and rest in ("attn_gate.weight", "ssm_alpha.weight"):
            continue
        else:
            yield blk + rest, a


def metadata(arch: str) -> dict:
    loop = arch == "sensenova_loop"
    blocks = len(visits()) + 1 if loop else QWEN35_LAYERS
    toks = tokens()
    m = {
        "general.architecture": arch,
        f"{arch}.embedding_length": H,
        f"{arch}.block_count": blocks,
        f"{arch}.attention.head_count": HEADS,
        f"{arch}.attention.head_count_kv": KV,
        f"{arch}.attention.key_length": HD,
        f"{arch}.attention.value_length": HD,
        f"{arch}.feed_forward_length": FFN,
        f"{arch}.context_length": 1024,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.full_attention_interval": INTERVAL,
        f"{arch}.rope.dimension_count": HD // 4,
        f"{arch}.rope.dimension_sections": [2, 1, 1, 0],
        f"{arch}.rope.freq_base": 10000000.0,
        f"{arch}.ssm.conv_kernel": CONV,
        f"{arch}.ssm.group_count": HK,
        f"{arch}.ssm.inner_size": INNER,
        f"{arch}.ssm.state_size": DK,
        f"{arch}.ssm.time_step_rank": HV,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": ["H e", "w o"],
        "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (len(toks) - 3),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }
    if loop:
        m.update({
            f"{arch}.nextn_predict_layers": 1,
            f"{arch}.attention.recurrent_layers":
                [_is_linear(i) for i in range(len(visits()))] + [False],
            f"{arch}.loop.start": START,
            f"{arch}.loop.end": END,
            f"{arch}.loop.count": PASSES,
            f"{arch}.mhc.start": START,
            f"{arch}.mhc.end": END,
            f"{arch}.mhc.streams": 2,
            f"{arch}.mhc.iterations": MHC_ITERATIONS,
            f"{arch}.mhc.fixed_identity": False,
            f"{arch}.mhc.pre_scale": 1.0,
        })
    return m


def write(path, arch: str, v: dict, *, fused: bool = False,
          q8: tuple[str, ...] = (), meta: dict | None = None) -> None:
    """Write ``v`` as an ``arch`` GGUF. Tensors whose file names end with a
    suffix in ``q8`` are written as Q8_0."""
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    meta = metadata(arch) if meta is None else meta
    w = GGUFWriter(str(path), arch)
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
    for name, a in _tensors(arch, v, fused):
        if q8 and name.endswith(q8):
            t = GGMLQuantizationType.Q8_0
            w.add_tensor(name, quantize(a, t), raw_dtype=t)
        else:
            w.add_tensor(name, np.ascontiguousarray(a))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
