"""A tiny random Nemotron speech encoder GGUF (arch ``pee``), with the real
file's tensor names, layouts and header keys.

Shared by the oracle script that writes the committed goldens
(``tiny_encoder.py``) and the CPU test that checks gmlx against them, so
both read the same file. Needs numpy and gguf only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

N_MELS = 128
N_FFT = 512
NUM_SPEAKERS = 8

# Small widths and a 20-frame chunk, so a 25 s clip spans many windows and
# overflows the 188-frame speaker cache.
FIELDS = {
    "pee.asr.feat_in": N_MELS, "pee.asr.d_model": 32, "pee.asr.n_heads": 2,
    "pee.asr.n_layers": 2, "pee.asr.ff_hidden": 64,
    "pee.asr.subsampling_factor": 8, "pee.asr.rotary_dims": 8,
    "pee.asr.qk_norm": True,
    "pee.diar.feat_in": N_MELS, "pee.diar.d_model": 16, "pee.diar.n_heads": 2,
    "pee.diar.n_layers": 2, "pee.diar.ff_hidden": 32,
    "pee.diar.subsampling_factor": 8, "pee.diar.rotary_dims": 8,
    "pee.diar.qk_norm": False, "pee.diar.num_speakers": NUM_SPEAKERS,
    "pee.diar.tf_d_model": 12, "pee.diar.upsample_factor": 8,
    "pee.fusion.speaker_activity_threshold": 0.5, "pee.fusion.proj_dim": 24,
    "pee.online.inference_length": 20, "pee.online.chunk_left_context": 3,
    "pee.online.chunk_right_context": 3,
    "pee.preprocessor.sample_rate": 16000,
    "pee.preprocessor.window_size": 0.025,
    "pee.preprocessor.window_stride": 0.01,
    "pee.preprocessor.n_fft": N_FFT, "pee.preprocessor.features": N_MELS,
    "ctc.d_model": 32, "ctc.n_heads": 2, "ctc.n_layers": 1,
    "ctc.ff_hidden": 16, "ctc.rotary_dims": 8, "ctc.vocab_size": 40,
    "ctc.upsampling_factor": 2, "ctc.residual_scale": 0.1,
    "ctc.qk_norm": True,
}


def speaker_embedding(speakers: int, d: int) -> np.ndarray:
    """The fixed sinusoidal table the real file stores as
    ``fusion.diar_kernel`` (transformers
    ``Nemotron3_5TranscribeSpeakerEmbedding``)."""
    pos = np.arange(speakers, dtype=np.float32)[:, None]
    inv = np.exp(np.arange(0, d, 2, dtype=np.float32)
                 * -(math.log(10000.0) / d)).astype(np.float32)
    out = np.zeros((speakers, d), np.float32)
    out[:, 0::2] = np.sin(pos * inv)
    out[:, 1::2] = np.cos(pos * inv)
    return out


def _layers(rng, prefix, n, d, ff, head, qk_norm, out, f16):
    for i in range(n):
        p = f"{prefix}.layers.{i}."
        out[p + "attn.w_qkv.weight"] = f16(rng.normal(0, d ** -0.5, (3 * d, d)))
        out[p + "attn.out_proj.weight"] = f16(rng.normal(0, d ** -0.5, (d, d)))
        out[p + "attn.out_proj.bias"] = rng.normal(0, 0.05, d)
        if qk_norm:
            for k in ("q_norm", "k_norm"):
                out[p + f"attn.{k}.weight"] = 1 + rng.normal(0, 0.1, head)
                out[p + f"attn.{k}.bias"] = rng.normal(0, 0.1, head)
        out[p + "ffn.net.0.weight"] = f16(rng.normal(0, d ** -0.5, (ff, d)))
        out[p + "ffn.net.0.bias"] = rng.normal(0, 0.05, ff)
        out[p + "ffn.net.3.weight"] = f16(rng.normal(0, ff ** -0.5, (d, ff)))
        out[p + "ffn.net.3.bias"] = rng.normal(0, 0.05, d)
        for k in ("norm1", "norm2"):
            out[p + f"{k}.weight"] = 1 + rng.normal(0, 0.1, d)
            out[p + f"{k}.bias"] = rng.normal(0, 0.1, d)
    for k in ("embed_norm", "final_norm"):
        out[f"{prefix}.{k}.weight"] = 1 + rng.normal(0, 0.1, d)
        out[f"{prefix}.{k}.bias"] = rng.normal(0, 0.1, d)


def tiny_weights(seed: int, mel_filters: np.ndarray) -> dict[str, np.ndarray]:
    """Random weights in numpy order: matrices float16, as the real file
    stores them, everything else float32. ``mel_filters`` is the
    ``(n_mels, n_fft // 2 + 1)`` bank stored as ``preprocessor.fb``."""
    rng = np.random.default_rng(seed)
    f = FIELDS
    out: dict[str, np.ndarray] = {}

    def f16(a):
        return np.asarray(a, np.float16)

    for b in ("asr", "diar"):
        d = f[f"pee.{b}.d_model"]
        sub = f[f"pee.{b}.subsampling_factor"]
        out[f"{b}.pre_encode.proj.weight"] = f16(
            rng.normal(0, (N_MELS * sub) ** -0.5, (d, N_MELS * sub)))
        _layers(rng, b, f[f"pee.{b}.n_layers"], d, f[f"pee.{b}.ff_hidden"],
                d // f[f"pee.{b}.n_heads"], f[f"pee.{b}.qk_norm"], out, f16)
    dd, hd = f["pee.diar.d_model"], f["pee.diar.tf_d_model"]
    up = f["pee.diar.upsample_factor"]
    h = "diar.head."
    out[h + "encoder_proj.weight"] = f16(rng.normal(0, dd ** -0.5, (hd, dd)))
    out[h + "encoder_proj.bias"] = rng.normal(0, 0.05, hd)
    out[h + "subpixel_upsample.weight"] = f16(
        rng.normal(0, (3 * hd) ** -0.5, (hd * up, hd, 3)))
    out[h + "subpixel_upsample.bias"] = rng.normal(0, 0.05, hd * up)
    out[h + "first_hidden_to_hidden.weight"] = f16(
        rng.normal(0, 2 * hd ** -0.5, (hd, hd)))
    out[h + "first_hidden_to_hidden.bias"] = rng.normal(0, 0.1, hd)
    # Wide logits around zero, so speakers switch on and off.
    out[h + "single_hidden_to_spks.weight"] = f16(
        rng.normal(0, 6 * hd ** -0.5, (NUM_SPEAKERS, hd)))
    out[h + "single_hidden_to_spks.bias"] = rng.normal(0, 1.0, NUM_SPEAKERS)
    out[h + "learnable_sil_emb"] = rng.normal(0, 1.0, dd)
    ad, pd = f["pee.asr.d_model"], f["pee.fusion.proj_dim"]
    out["fusion.asr_norm.weight"] = 1 + rng.normal(0, 0.1, ad)
    out["fusion.asr_norm.bias"] = rng.normal(0, 0.1, ad)
    out["fusion.diar_norm.weight"] = 1 + rng.normal(0, 0.1, NUM_SPEAKERS)
    out["fusion.diar_norm.bias"] = rng.normal(0, 0.1, NUM_SPEAKERS)
    out["fusion.diar_kernel"] = speaker_embedding(NUM_SPEAKERS, ad)
    out["fusion.proj.weight"] = f16(rng.normal(0, ad ** -0.5, (pd, ad)))
    out["fusion.proj.bias"] = rng.normal(0, 0.05, pd)
    cd, u = f["ctc.d_model"], f["ctc.upsampling_factor"]
    # The ConvTranspose1d as a Linear: rows j * D + o, the bias repeated.
    out["ctc.upsample.weight"] = rng.normal(0, cd ** -0.5, (u * cd, cd))
    out["ctc.upsample.bias"] = np.tile(rng.normal(0, 0.05, cd), u)
    _layers(rng, "ctc.transformer", f["ctc.n_layers"], cd, f["ctc.ff_hidden"],
            cd // f["ctc.n_heads"], True, out, np.asarray)
    out["ctc.classifier.weight"] = rng.normal(0, 3 * cd ** -0.5,
                                              (f["ctc.vocab_size"], cd))
    out["ctc.classifier.bias"] = rng.normal(0, 0.5, f["ctc.vocab_size"])
    out["preprocessor.fb"] = np.asarray(mel_filters, np.float32)
    return {k: (v if v.dtype == np.float16 else np.asarray(v, np.float32))
            for k, v in out.items()}


def write_gguf(path: str | Path, weights: dict[str, np.ndarray]) -> Path:
    """Write ``weights`` and :data:`FIELDS` as a ``pee`` GGUF."""
    import gguf

    path = Path(path)
    w = gguf.GGUFWriter(str(path), "pee")
    for k, v in FIELDS.items():
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        else:
            w.add_float32(k, float(v))
    for name in sorted(weights):
        w.add_tensor(name, np.ascontiguousarray(weights[name]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
