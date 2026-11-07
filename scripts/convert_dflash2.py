#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# /// script
# requires-python = ">=3.11.1"
# dependencies = ["mlx-kquant>=0.5.9,<0.6", "gguf>=0.19", "numpy"]
# ///
"""Convert a DFlash2 drafter checkpoint to a llama.cpp ``dflash`` GGUF.

The source is a Hugging Face DFlash2 checkpoint (``DFlash2DraftModel``) in
safetensors, such as ricursiveAI/Ripple-65B-Preview-DFlash2. The GGUF has
llama.cpp's DFlash key and tensor set, and the tokenizer of the target GGUF,
so llama.cpp and gmlx both load it next to that target.

Two keys go beyond what llama.cpp's own converter writes:

  dflash.attention.rope_pattern    false for a layer in no_rope_layers
  dflash.attention.context_window  the context rows a full layer sees

gmlx reads both. llama.cpp ignores them, so it ropes every layer and lets
the full layers see the whole context.

Usage:
  uv run scripts/convert_dflash2.py \\
      --checkpoint ~/llm/hf/ricursiveAI__Ripple-65B-Preview-DFlash2 \\
      --target-gguf Ripple-65B-Preview-AD-Q4_K_M.gguf \\
      --outtype q8_0 --out Ripple-65B-Preview-DFlash2-Q8_0.gguf

``--outtype`` is bf16, q8_0, q4_0 or q4_k_m. Q8_0 and Q4_0 bytes come from
gguf-py, which matches llama-quantize. The K-quants come from mlx-kquant,
so the script needs Apple Silicon for q4_k_m.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as VT

# The context rows a full layer was trained to see, by target model name.
# Ripple: inference/server/model.py DRAFT_TRAINING_CONTEXT (12,288 context
# plus 4,096 predicted rows).
CONTEXT_WINDOWS = {
    "Ripple-65B-Preview": 16384,
}

OUTTYPES = {
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
}

BLK = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
}
ROOT = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}
# Tensors that stay F32 whatever the outtype, besides every 1-D tensor.
F32_STEMS = ("attn_conv_base", "ffn_conv_base")


def die(msg: str) -> None:
    sys.exit(f"convert_dflash2: {msg}")


# --- safetensors -------------------------------------------------------------

class Safetensors:
    """Reads tensors from one or more safetensors files through mmap. numpy
    has no bfloat16, so BF16 is widened to float32 by hand."""

    def __init__(self, folder: Path):
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            paths = [folder / f for f in files]
        else:
            paths = sorted(folder.glob("*.safetensors"))
        if not paths:
            die(f"no safetensors files in {folder}")
        self.entries: dict[str, tuple[mmap.mmap, int, dict]] = {}
        self._maps = []
        for path in paths:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps.append(mm)
            (n,) = struct.unpack("<Q", mm[:8])
            header = json.loads(mm[8:8 + n])
            header.pop("__metadata__", None)
            for name, info in header.items():
                if name in self.entries:
                    die(f"tensor {name} is in two files")
                self.entries[name] = (mm, 8 + n, info)

    def names(self):
        return list(self.entries)

    def shape(self, name: str) -> list[int]:
        return list(self.entries[name][2]["shape"])

    def get(self, name: str) -> np.ndarray:
        mm, base, info = self.entries[name]
        start, end = info["data_offsets"]
        raw = np.frombuffer(mm, dtype=np.uint8, count=end - start,
                            offset=base + start)
        dtype, shape = info["dtype"], info["shape"]
        if dtype == "BF16":
            bits = raw.view(np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(shape)
        if dtype == "F16":
            return raw.view(np.float16).astype(np.float32).reshape(shape)
        if dtype == "F32":
            return raw.view(np.float32).reshape(shape).copy()
        die(f"tensor {name} has unsupported dtype {dtype}")
        raise AssertionError


# --- type policy -------------------------------------------------------------

def use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama-quantize's rule for the layers a _M mix gives more bits."""
    return (i_layer < n_layers // 8 or i_layer >= 7 * n_layers // 8
            or (i_layer - n_layers // 8) % 3 == 2)


def tensor_type(name: str, shape: list[int], outtype: str, n_layers: int) -> Q:
    if len(shape) == 1 or name.split(".")[-1] in F32_STEMS:
        return Q.F32
    if outtype == "bf16":
        return Q.BF16
    if outtype == "q8_0":
        return Q.Q8_0
    if outtype == "q4_0":
        return Q.Q4_0
    if name.startswith("blk.") and name.endswith(
            ("attn_v.weight", "ffn_down.weight")):
        if use_more_bits(int(name.split(".")[1]), n_layers):
            return Q.Q6_K
    return Q.Q4_K


def encode(data: np.ndarray, qtype: Q) -> np.ndarray:
    """Wire bytes (or F32/BF16 values) for one tensor."""
    if qtype == Q.F32:
        return np.ascontiguousarray(data, dtype=np.float32)
    if qtype == Q.BF16:
        bits = np.ascontiguousarray(data, dtype=np.float32).view(np.uint32)
        # Round to nearest even, as gguf-py and llama.cpp do.
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return rounded.astype(np.uint16)
    if qtype in (Q.Q8_0, Q.Q4_0):
        return gguf.quants.quantize(np.ascontiguousarray(data), qtype)
    import mlx.core as mx
    import mlx_kquant as kq

    codec = {Q.Q4_K: "q4_k", Q.Q6_K: "q6_k"}[qtype]
    wq, _ = kq.quantize(mx.array(data), codec)
    out = np.array(wq)
    return out.reshape(data.shape[:-1] + (-1,))


def info_args(shape: list[int], qtype: Q):
    """(shape, numpy dtype, nbytes) for GGUFWriter.add_tensor_info."""
    n = int(np.prod(shape))
    if qtype == Q.F32:
        return shape, np.dtype(np.float32), 4 * n
    if qtype == Q.BF16:
        return shape, np.dtype(np.uint16), 2 * n
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    if shape[-1] % block:
        die(f"row length {shape[-1]} is not a multiple of {block} for {qtype.name}")
    row_bytes = shape[-1] // block * size
    return shape[:-1] + [row_bytes], np.dtype(np.uint8), n // block * size


# --- header ------------------------------------------------------------------

def read_target(path: Path):
    reader = gguf.GGUFReader(path)
    fields = reader.fields
    arch = fields["general.architecture"].contents()

    def get(key, default=None):
        f = fields.get(key)
        return default if f is None else f.contents()

    return reader, {
        "arch": arch,
        "name": get("general.name"),
        "license": get("general.license"),
        "n_layers": get(f"{arch}.block_count"),
        "hidden": get(f"{arch}.embedding_length"),
        "n_vocab": len(fields["tokenizer.ggml.tokens"].data),
    }


def copy_tokenizer(writer: gguf.GGUFWriter, reader: gguf.GGUFReader) -> None:
    """Every tokenizer key of the target except its mask token, so llama.cpp's
    draft and target vocab check passes."""
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer.") or key == "tokenizer.ggml.mask_token_id":
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == VT.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub)


def card_base_model(folder: Path) -> str | None:
    """The first ``base_model`` repo id in the checkpoint's README front
    matter, such as ``ricursiveAI/Ripple-65B-Preview``."""
    readme = folder / "README.md"
    if not readme.is_file():
        return None
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            break
        if line.startswith("base_model:"):
            value = line.split(":", 1)[1].strip()
            if not value and i + 1 < len(lines):
                value = lines[i + 1].strip().lstrip("-").strip()
            return value.strip("'\"") or None
    return None


def size_label(n_params: int) -> str:
    return f"{n_params / 1e9:.1f}B" if n_params >= 1e9 else f"{n_params / 1e6:.0f}M"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="folder with config.json and the safetensors")
    ap.add_argument("--target-gguf", type=Path, required=True,
                    help="any GGUF of the target model (tokenizer and checks)")
    ap.add_argument("--outtype", choices=sorted(OUTTYPES), default="q8_0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-window", type=int, default=None,
                    help="context rows a full layer sees; 0 for none. "
                         "Defaults to the known value for the target")
    args = ap.parse_args(argv)

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if "DFlash2DraftModel" not in (cfg.get("architectures") or []):
        die("config.json does not describe a DFlash2DraftModel")
    dcfg = cfg.get("dflash_config") or {}
    n_layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg.get("head_dim") or hidden // int(cfg["num_attention_heads"]))
    layer_types = cfg.get("layer_types") or ["full_attention"] * n_layers
    if len(layer_types) != n_layers:
        die(f"layer_types has {len(layer_types)} entries for {n_layers} layers")
    no_rope = set(int(i) for i in cfg.get("no_rope_layers") or ())
    if any(not 0 <= i < n_layers for i in no_rope):
        die(f"no_rope_layers {sorted(no_rope)} is out of range")
    targets = [int(i) for i in dcfg.get("target_layer_ids") or ()]
    if not targets:
        die("dflash_config.target_layer_ids is missing")

    reader, target = read_target(args.target_gguf)
    if target["hidden"] != hidden:
        die(f"drafter width {hidden} != target embedding_length {target['hidden']}")
    if max(targets) + 1 > int(target["n_layers"]):
        die(f"target_layer_ids {targets} exceed the target's {target['n_layers']} layers")

    has_full = any(t != "sliding_attention" for t in layer_types)
    context = args.context_window
    if context is None and has_full:
        context = CONTEXT_WINDOWS.get(target["name"])
        if context is None:
            die(f"no known context window for target {target['name']!r}; "
                "pass --context-window (0 for none)")

    st = Safetensors(args.checkpoint)
    plan = []
    for src in st.names():
        src_key = src.removeprefix("model.")
        if src_key.startswith("layers."):
            _, idx, leaf = src_key.split(".", 2)
            dst = BLK.get(leaf)
            dst = None if dst is None else f"blk.{idx}.{dst}"
        else:
            dst = ROOT.get(src_key)
        if dst is None:
            die(f"unknown tensor {src}")
        plan.append((dst, src))
    names = {dst for dst, _ in plan}
    want = set(ROOT.values()) | {f"blk.{i}.{v}" for i in range(n_layers)
                                 for v in BLK.values()}
    if names != want:
        die(f"missing tensors: {sorted(want - names)}")
    for cb in ("candidate_selector.predecessor_codebook",
               "candidate_selector.successor_codebook"):
        rows = st.shape(cb)[0]
        if rows != target["n_vocab"]:
            die(f"{cb} has {rows} rows, the target vocab is {target['n_vocab']}")

    writer = gguf.GGUFWriter(args.out, "dflash")
    arch = "dflash"
    n_params = sum(int(np.prod(st.shape(src))) for _, src in plan)
    writer.add_type("model")
    writer.add_name(f"{target['name']}-DFlash2")
    writer.add_basename(target["name"])
    writer.add_finetune("DFlash2")
    writer.add_size_label(size_label(n_params))
    if target["license"]:
        writer.add_license(target["license"])
    writer.add_base_model_count(1)
    writer.add_base_model_name(0, target["name"])
    base_repo = card_base_model(args.checkpoint)
    if base_repo and "/" in base_repo:
        writer.add_base_model_organization(0, base_repo.split("/", 1)[0])
        writer.add_base_model_repo_url(0, f"https://huggingface.co/{base_repo}")

    writer.add_uint32(f"{arch}.block_count", n_layers)
    writer.add_uint32(f"{arch}.context_length", int(cfg["max_position_embeddings"]))
    writer.add_uint32(f"{arch}.embedding_length", hidden)
    writer.add_uint32(f"{arch}.feed_forward_length", int(cfg["intermediate_size"]))
    writer.add_uint32(f"{arch}.attention.head_count", int(cfg["num_attention_heads"]))
    writer.add_uint32(f"{arch}.attention.head_count_kv", int(cfg["num_key_value_heads"]))
    writer.add_bool(f"{arch}.attention.causal", bool(cfg.get("is_causal", False)))
    rope = cfg.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        die(f"rope_type {rope['rope_type']!r} is not supported")
    writer.add_float32(f"{arch}.rope.freq_base",
                       float(rope.get("rope_theta", cfg.get("rope_theta"))))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon",
                       float(cfg["rms_norm_eps"]))
    writer.add_uint32(f"{arch}.attention.key_length", head_dim)
    writer.add_uint32(f"{arch}.attention.value_length", head_dim)
    writer.add_uint32(f"{arch}.block_size", int(dcfg.get("block_size", 16)))
    for key in ("conv_kernel_size", "conv_group_size", "selector_rank",
                "selector_top_k"):
        writer.add_uint32(f"{arch}.{key}", int(dcfg[key]))
    writer.add_array(f"{arch}.target_layers", [i + 1 for i in targets])
    if cfg.get("use_sliding_window") and cfg.get("sliding_window"):
        writer.add_uint32(f"{arch}.attention.sliding_window", int(cfg["sliding_window"]))
        writer.add_array(f"{arch}.attention.sliding_window_pattern",
                         [t == "sliding_attention" for t in layer_types])
    if no_rope:
        writer.add_array(f"{arch}.attention.rope_pattern",
                         [i not in no_rope for i in range(n_layers)])
    if context:
        writer.add_uint32(f"{arch}.attention.context_window", int(context))

    copy_tokenizer(writer, reader)
    mask = dcfg.get("mask_token_id")
    if mask is None:
        die("dflash_config.mask_token_id is missing")
    writer.add_mask_token_id(int(mask))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(OUTTYPES[args.outtype])

    types = []
    for dst, src in plan:
        shape = st.shape(src)
        qtype = tensor_type(dst, shape, args.outtype, n_layers)
        types.append(qtype)
        info_shape, dtype, nbytes = info_args(shape, qtype)
        writer.add_tensor_info(dst, info_shape, dtype, nbytes, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for (dst, src), qtype in zip(plan, types):
        print(f"{dst:40s} {qtype.name:5s} {st.shape(src)}", flush=True)
        writer.write_tensor_data(encode(st.get(src), qtype))
    writer.close()
    print(f"wrote {args.out} ({len(plan)} tensors, {args.outtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
