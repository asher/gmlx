#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash DSpark (MTP) shards to a deepseek4-dspark sidecar GGUF.

The upstream HF repo isolates the DSpark drafter in its final shards (mtp.* tensors
only). This script reads those safetensors shards plus the inference config and
emits a companion GGUF matching the deepseek4-dspark layout (arch, kv keys and
tensor names as in antirez/deepseek-v4-gguf's DSpark-support sidecar), which gmlx
auto-discovers next to a deepseek4 target GGUF.

Routed experts are stored upstream as packed fp4 (e2m1 pairs) with ue8m0 scales
per 32 elements - exactly the GGML MXFP4 block format - so with the default
--experts-codec mxfp4 they are repacked losslessly (bit-identical dequant).
Attention/dense fp8(e4m3, 128x128 block scales) tensors are dequantized and
re-encoded as Q8_0; norms, sinks, router and hyper-connection params stay F32;
the markov heads are written F16.

Usage:
  python scripts/convert_dspark_sidecar.py \
      --shards-dir ~/llm/hf/DeepSeek-V4-Flash-0731-mtp \
      --out ~/llm/gguf/.../DeepSeek-V4-Flash-0731-DSpark-MXFP4-Q8_0-F32.gguf
"""

from __future__ import annotations

import argparse
import glob
import json
import mmap
import os
import struct
import sys

import numpy as np

import gguf
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize as ggml_quantize, dequantize as ggml_dequantize

Q = GGMLQuantizationType


def _build_fp8_e4m3_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0x0F and mant == 0x07:
            val = np.nan
        elif exp == 0:
            val = (mant / 8.0) * 2.0 ** (-6)
        else:
            val = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        lut[b] = sign * val
    return lut


_FP8_LUT = _build_fp8_e4m3_lut()


def _e8m0_to_f32(e: np.ndarray) -> np.ndarray:
    return np.float32(2.0) ** (e.astype(np.int32) - 127)


class ShardReader:
    """Byte-range reader over a set of safetensors files."""

    def __init__(self, paths: list[str]):
        self.entries: dict[str, tuple[str, dict]] = {}
        self._maps: dict[str, tuple[mmap.mmap, int]] = {}
        for p in paths:
            with open(p, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
            header.pop("__metadata__", None)
            for name, info in header.items():
                self.entries[name] = (p, info)
            f = open(p, "rb")
            self._maps[p] = (mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), 8 + hlen)

    def raw(self, name: str) -> tuple[np.ndarray, str, list[int]]:
        path, info = self.entries[name]
        m, data_start = self._maps[path]
        a, b = info["data_offsets"]
        buf = np.frombuffer(m, dtype=np.uint8, count=b - a, offset=data_start + a)
        return buf, info["dtype"], info["shape"]

    def f32(self, name: str) -> np.ndarray:
        """Tensor dequantized/cast to float32 (handles BF16, F32, F8_E4M3+scale)."""
        buf, dtype, shape = self.raw(name)
        if dtype == "F32":
            return buf.view(np.float32).reshape(shape).copy()
        if dtype == "BF16":
            u32 = buf.view(np.uint16).astype(np.uint32) << 16
            return u32.view(np.float32).reshape(shape)
        if dtype == "F8_E4M3":
            w = _FP8_LUT[buf].reshape(shape)
            sbuf, sdtype, sshape = self.raw(name.replace(".weight", ".scale"))
            assert sdtype == "F8_E8M0", f"{name}: unexpected scale dtype {sdtype}"
            scale = _e8m0_to_f32(sbuf).reshape(sshape)
            out, inn = shape
            scale = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            return w * scale[:out, :inn]
        raise ValueError(f"{name}: unhandled dtype {dtype}")


def repack_fp4_to_mxfp4(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Losslessly repack upstream packed-fp4 rows into GGML MXFP4 blocks.

    weight_u8: [rows, in/2] with element 2j in the low nibble of byte j.
    scale_u8:  [rows, in/32] ue8m0 bytes.
    Returns [rows, in/32, 17] block bytes (1 scale byte + 16 code bytes, where
    GGML wants elements 0..15 in low nibbles and 16..31 in high nibbles).
    """
    rows, half = weight_u8.shape
    n = half * 2
    codes = np.empty((rows, n), dtype=np.uint8)
    codes[:, 0::2] = weight_u8 & 0x0F
    codes[:, 1::2] = weight_u8 >> 4
    codes = codes.reshape(rows, n // 32, 32)
    qs = codes[:, :, :16] | (codes[:, :, 16:] << np.uint8(4))
    return np.concatenate([scale_u8[:, :, None], qs], axis=2)


def _verify_mxfp4_repack(blocks: np.ndarray, weight_u8: np.ndarray, scale_u8: np.ndarray) -> None:
    """Cross-check one row of repacked blocks against the reference fp4 dequant."""
    fp4_table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
    got = ggml_dequantize(blocks[0].reshape(1, -1), Q.MXFP4).reshape(-1)
    codes = np.empty(weight_u8.shape[1] * 2, dtype=np.uint8)
    codes[0::2] = weight_u8[0] & 0x0F
    codes[1::2] = weight_u8[0] >> 4
    ref = fp4_table[codes] * np.repeat(_e8m0_to_f32(scale_u8[0]), 32)
    if not np.array_equal(got, ref):
        raise AssertionError("MXFP4 repack does not match reference fp4 dequant")


class Plan:
    """One output tensor: gguf name, logical shape, codec, and a producer."""

    def __init__(self, name, shape, codec, produce):
        self.name = name
        self.shape = tuple(shape)
        self.codec = codec
        self.produce = produce
        block, type_size = gguf.GGML_QUANT_SIZES[codec]
        assert self.shape[-1] % block == 0, (name, self.shape, codec)
        # gguf's add_tensor_info wants the encoded byte-shape for quantized
        # codecs and the element shape for float types.
        self.byte_shape = self.shape[:-1] + (self.shape[-1] // block * type_size,)
        self.nbytes = int(np.prod(self.byte_shape))
        if codec in (Q.F32, Q.F16):
            self.info_shape, self.info_dtype, self.info_raw = (
                self.shape, np.dtype(np.float32 if codec == Q.F32 else np.float16), None)
        else:
            self.info_shape, self.info_dtype, self.info_raw = (
                self.byte_shape, np.dtype(np.uint8), codec)


def build_plan(reader: ShardReader, cfg: dict, experts_codec: Q) -> list[Plan]:
    n_stages = cfg["n_mtp_layers"]
    n_experts = cfg["n_routed_experts"]
    plans: list[Plan] = []

    def enc(src: str, codec: Q):
        def produce():
            data = reader.f32(src)
            if codec == Q.F32:
                return np.ascontiguousarray(data, dtype=np.float32)
            if codec == Q.F16:
                return data.astype(np.float16)
            return ggml_quantize(data, codec)
        return produce

    def dense(dst: str, src: str, codec: Q):
        shape = reader.entries[src][1]["shape"]
        plans.append(Plan(dst, shape, codec, enc(src, codec)))

    def experts(dst: str, w: str, k: int):
        src0 = f"mtp.{k}.ffn.experts.0.{w}.weight"
        rows, half = reader.entries[src0][1]["shape"]
        shape = (n_experts, rows, half * 2)

        def produce():
            out = np.empty((n_experts, rows, (half * 2) // 32, 17), dtype=np.uint8)
            for i in range(n_experts):
                wbuf, wd, wshape = reader.raw(f"mtp.{k}.ffn.experts.{i}.{w}.weight")
                sbuf, sd, sshape = reader.raw(f"mtp.{k}.ffn.experts.{i}.{w}.scale")
                assert wd == "I8" and sd == "F8_E8M0", (wd, sd)
                wu8 = wbuf.view(np.uint8).reshape(wshape)
                su8 = sbuf.reshape(sshape)
                if experts_codec == Q.MXFP4:
                    blocks = repack_fp4_to_mxfp4(wu8, su8)
                    if i == 0:
                        _verify_mxfp4_repack(blocks, wu8, su8)
                    out[i] = blocks
                else:
                    fp4_table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                                         dtype=np.float32)
                    codes = np.empty((rows, half * 2), dtype=np.uint8)
                    codes[:, 0::2] = wu8 & 0x0F
                    codes[:, 1::2] = wu8 >> 4
                    f = fp4_table[codes] * np.repeat(_e8m0_to_f32(su8), 32, axis=1)
                    # gguf-py has no k-quant encoders; mlx_kquant does.
                    import mlx.core as mx
                    import mlx_kquant as kq

                    wq, _ = kq.quantize(mx.array(f), experts_codec.name.lower())
                    out2 = np.asarray(wq).reshape(rows, -1)
                    if i == 0:
                        rt = ggml_dequantize(out2, experts_codec)
                        rms = float(np.sqrt(np.mean((rt - f) ** 2)))
                        ref = float(np.sqrt(np.mean(f**2)))
                        print(f"    {dst}: {experts_codec.name} round-trip rms "
                              f"{rms:.4g} (weight rms {ref:.4g})")
                        nonlocal_shape[0] = (n_experts,) + out2.shape
                        out_l.append(np.empty((n_experts,) + out2.shape, dtype=np.uint8))
                    out_l[0][i] = out2
            if experts_codec == Q.MXFP4:
                return out.reshape(n_experts, rows, -1)
            return out_l[0]

        nonlocal_shape = [None]
        out_l: list[np.ndarray] = []
        plans.append(Plan(dst, shape, experts_codec, produce))

    for k in range(n_stages):
        p = f"mtp.{k}."
        dense(p + "attn_q_a.weight", p + "attn.wq_a.weight", Q.Q8_0)
        dense(p + "attn_q_a_norm.weight", p + "attn.q_norm.weight", Q.F32)
        dense(p + "attn_q_b.weight", p + "attn.wq_b.weight", Q.Q8_0)
        dense(p + "attn_kv.weight", p + "attn.wkv.weight", Q.Q8_0)
        dense(p + "attn_kv_a_norm.weight", p + "attn.kv_norm.weight", Q.F32)
        dense(p + "attn_output_a.weight", p + "attn.wo_a.weight", Q.Q8_0)
        dense(p + "attn_output_b.weight", p + "attn.wo_b.weight", Q.Q8_0)
        dense(p + "attn_norm.weight", p + "attn_norm.weight", Q.F32)
        dense(p + "attn_sinks.weight", p + "attn.attn_sink", Q.F32)
        dense(p + "ffn_norm.weight", p + "ffn_norm.weight", Q.F32)
        dense(p + "ffn_gate_inp.weight", p + "ffn.gate.weight", Q.F32)
        dense(p + "exp_probs_b.bias", p + "ffn.gate.bias", Q.F32)
        experts(p + "ffn_gate_exps.weight", "w1", k)
        experts(p + "ffn_up_exps.weight", "w3", k)
        experts(p + "ffn_down_exps.weight", "w2", k)
        dense(p + "ffn_gate_shexp.weight", p + "ffn.shared_experts.w1.weight", Q.Q8_0)
        dense(p + "ffn_up_shexp.weight", p + "ffn.shared_experts.w3.weight", Q.Q8_0)
        dense(p + "ffn_down_shexp.weight", p + "ffn.shared_experts.w2.weight", Q.Q8_0)
        for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            dense(p + hc + ".weight", p + hc, Q.F32)
        if k == 0:
            dense(p + "main_proj.weight", p + "main_proj.weight", Q.Q8_0)
            dense(p + "main_norm.weight", p + "main_norm.weight", Q.F32)
        if k == n_stages - 1:
            for hc in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                dense(p + hc + ".weight", p + hc, Q.F32)
            dense(p + "norm.weight", p + "norm.weight", Q.F32)
            dense(p + "markov_head.markov_w1.weight", p + "markov_head.markov_w1.weight", Q.F16)
            dense(p + "markov_head.markov_w2.weight", p + "markov_head.markov_w2.weight", Q.F16)
            dense(p + "confidence_head.proj.weight", p + "confidence_head.proj.weight", Q.F32)
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards-dir", required=True,
                    help="Directory with the mtp.* safetensors shards and inference config.json")
    ap.add_argument("--out", required=True, help="Output sidecar .gguf path")
    ap.add_argument("--config", default=None,
                    help="Inference-style config json (default: <shards-dir>/config.json)")
    ap.add_argument("--experts-codec", default="mxfp4",
                    choices=["mxfp4", "q4_k", "q3_k", "q2_k"],
                    help="Codec for routed experts; mxfp4 is a lossless repack of the fp4 weights")
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shards_dir, "*.safetensors")))
    if not shard_paths:
        sys.exit(f"no safetensors shards in {args.shards_dir}")
    cfg_path = args.config or os.path.join(args.shards_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for key in ("n_mtp_layers", "dspark_block_size", "dspark_noise_token_id",
                "dspark_target_layer_ids", "dspark_markov_rank", "n_routed_experts"):
        if key not in cfg:
            sys.exit(f"{cfg_path}: missing {key} (need the inference/config.json, not the HF one)")

    reader = ShardReader(shard_paths)
    mtp_names = [n for n in reader.entries if n.startswith("mtp.")]
    if not mtp_names:
        sys.exit("shards contain no mtp.* tensors")
    experts_codec = {"mxfp4": Q.MXFP4, "q4_k": Q.Q4_K, "q3_k": Q.Q3_K, "q2_k": Q.Q2_K}[args.experts_codec]

    plans = build_plan(reader, cfg, experts_codec)

    consumed = set()
    for name in mtp_names:
        consumed.add(name)
    produced = {p.name for p in plans}
    print(f"{len(reader.entries)} source tensors -> {len(produced)} sidecar tensors")

    w = GGUFWriter(args.out, "deepseek4-dspark")
    w.add_name("DeepSeek V4 Flash 0731 DSpark support")
    w.add_uint32("dspark.block_size", cfg["dspark_block_size"])
    w.add_uint32("dspark.markov_rank", cfg["dspark_markov_rank"])
    w.add_uint32("dspark.n_layers", cfg["n_mtp_layers"])
    w.add_uint32("dspark.noise_token_id", cfg["dspark_noise_token_id"])
    w.add_uint32("dspark.stage_count", cfg["n_mtp_layers"])
    w.add_array("dspark.target_layer_ids", [int(i) for i in cfg["dspark_target_layer_ids"]])

    for p in plans:
        w.add_tensor_info(p.name, p.info_shape, p.info_dtype, p.nbytes, raw_dtype=p.info_raw)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, p in enumerate(plans):
        data = p.produce()
        assert data.nbytes == p.nbytes, (p.name, data.nbytes, p.nbytes)
        w.write_tensor_data(data)
        print(f"[{i + 1}/{len(plans)}] {p.name} {p.shape} {p.codec.name} {p.nbytes / 1e6:.1f}MB")
    w.close()
    total = os.path.getsize(args.out)
    print(f"wrote {args.out} ({total / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
