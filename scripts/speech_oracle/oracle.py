"""Oracle-side helpers for Nemotron 3.5 Transcribe, imported only by oracle
scripts in the speech oracle environment: the offline guard, the torch
setup, the encoder GGUF read into the transformers reference model, and
the fixture writer.

The reference is transformers' ``nemotron3_5_transcribe`` at
``TRANSFORMERS_REV``. Its audio tower, projector and word timestamp head
load from the encoder GGUF, through the tensor names NeMo-Speech.cpp's
``conversion/pee.py`` and ``conversion/ctc_timestamp.py`` write and the
``nemo_speechlm`` renames of transformers' ``conversion_mapping.py``. The
language model is a one-layer stand-in that never runs.

The oracle computes on the CPU. ``fp32`` runs every module in float32,
``bf16`` in bfloat16; the gap between the two calibrates each tolerance.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np

TRANSFORMERS_REV = "9b283429f9bf5c680bec437ab888c435dd2fef8b"
MODES = ("fp32", "bf16")
PACKAGES = ("torch", "torchaudio", "transformers", "tokenizers", "gguf",
            "sentencepiece", "librosa", "scipy", "numpy")

# NeMo ParallelExpertEncoder defaults that the checkpoint config omits
# (transformers convert_nemotron3_5_transcribe_to_hf.py
# NEMO_ONLINE_INFERENCE_DEFAULTS), in encoder frames.
FIFO_LENGTH = 40
SPEAKER_CACHE_UPDATE_PERIOD = 300
SPEAKER_CACHE_LENGTH = 188

# The checkpoint's sortformer_modules values (hf/config.json).
SPEAKER_CACHE_POLICY = {
    "speaker_cache_silence_frames_per_speaker": 1,
    "prediction_score_threshold": 0.25,
    "latest_frames_score_boost": 0.05,
    "strong_boost_rate": 0.75,
    "weak_boost_rate": 1.5,
    "min_positive_scores_rate": 0.5,
}

# processor_config.json feature_extractor.
FEATURES = {"feature_size": 128, "hop_length": 160, "n_fft": 512,
            "win_length": 400, "preemphasis": 0.97, "sampling_rate": 16000,
            "mel_filters_dtype": "bfloat16", "padding_value": 0.0,
            "return_attention_mask": True}
SUBSAMPLING = 8


def require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("error: oracle runs need HF_HUB_OFFLINE=1. Start "
                         "them with scripts/speech_oracle/run.py")


def setup(threads: int | None = None):
    """Import torch for an oracle run on the CPU and return it."""
    require_offline()
    import torch

    n = threads or int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if n:
        torch.set_num_threads(n)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    return torch


def torch_dtype(mode: str):
    import torch

    return {"fp32": torch.float32, "bf16": torch.bfloat16}[mode]


def fixtures_module():
    """scripts/image_oracle/fixtures.py: the fixture writer and reader."""
    path = Path(__file__).resolve().parent.parent / "image_oracle" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("_oracle_fixtures", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_dir() -> Path:
    raw = os.environ.get("GMLX_SPEECH_FIXTURES")
    if not raw:
        raise SystemExit("error: set GMLX_SPEECH_FIXTURES (run.py does)")
    return Path(raw)


def writer(name: str, *, check: str, script: str):
    fx = fixtures_module()
    w = fx.FixtureWriter(name, check=check, family="nemotron-transcribe",
                         script=script)
    w.manifest["oracle"] = {**fx.environment(),
                            "packages": fx.package_versions(PACKAGES),
                            "transformers_rev": TRANSFORMERS_REV}
    return w


def stable_speaker_cache_topk() -> None:
    """Make the reference speaker cache break score ties by the lowest
    index, as gmlx does. A frame the cache keeps for two speakers is
    stored twice, so later compressions see exact ties, which
    ``torch.topk`` orders as its sort happens to; gmlx's order equals a
    stable sort's."""
    import torch
    from transformers.models.nemotron3_diarization import \
        modeling_nemotron3_diarization as m

    def topk(x, k, dim=-1, largest=True, sorted=True):  # noqa: A002
        vals, idx = torch.sort(x, dim=dim, descending=largest, stable=True)
        return vals.narrow(dim, 0, k), idx.narrow(dim, 0, k)

    cls = m.Nemotron3DiarizationSpeakerCache
    for name in ("_boost_scores", "_compress"):
        def wrapped(self, *args, _f=getattr(cls, name), **kw):
            real = torch.topk
            torch.topk = topk
            try:
                return _f(self, *args, **kw)
            finally:
                torch.topk = real
        setattr(cls, name, wrapped)


# -- the encoder GGUF ---------------------------------------------------------


def read_gguf(path: str | os.PathLike) -> tuple[dict, dict]:
    """(metadata, tensors) of the encoder GGUF; tensors as float32 numpy
    arrays in numpy order (ggml's dimensions reversed)."""
    import gguf

    r = gguf.GGUFReader(str(path))
    meta = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            meta[k] = f.contents()
        except Exception:  # noqa: BLE001 - a field gguf cannot decode
            continue
    tensors = {}
    for t in r.tensors:
        if t.tensor_type.name not in ("F32", "F16"):
            raise SystemExit(f"error: {t.name} is {t.tensor_type.name}; the "
                             "oracle reads F32 and F16 encoders only")
        shape = tuple(int(d) for d in reversed(t.shape))
        tensors[t.name] = np.asarray(t.data, dtype=np.float32).reshape(shape)
    return meta, tensors


_INNER = [
    (r"\.pre_encode\.proj\.", ".embedder.projection."),
    (r"\.embed_norm\.", ".input_layer_norm."),
    (r"\.final_norm\.", ".layer_norm."),
    (r"\.norm1\.", ".layer_norm1."),
    (r"\.norm2\.", ".layer_norm2."),
    (r"\.attn\.out_proj\.", ".self_attn.o_proj."),
    (r"\.attn\.q_norm\.", ".self_attn.q_norm."),
    (r"\.attn\.k_norm\.", ".self_attn.k_norm."),
    (r"\.ffn\.net\.0\.", ".mlp.fc1."),
    (r"\.ffn\.net\.3\.", ".mlp.fc2."),
]
_DIAR = "model.audio_tower.diarization_model."
_OUTER = [
    (r"^asr\.", "model.audio_tower.asr_encoder."),
    (r"^diar\.head\.encoder_proj\.", _DIAR + "model.proj."),
    (r"^diar\.head\.subpixel_upsample\.", _DIAR + "model.upsampler.conv."),
    (r"^diar\.head\.first_hidden_to_hidden\.", _DIAR + "classifier.dense."),
    (r"^diar\.head\.single_hidden_to_spks\.", _DIAR + "classifier.out_proj."),
    (r"^diar\.head\.learnable_sil_emb$", _DIAR + "silence_embeds"),
    (r"^diar\.", _DIAR + "model.audio_tower."),
    (r"^fusion\.asr_norm\.", "model.audio_tower.asr_norm."),
    (r"^fusion\.diar_norm\.", "model.audio_tower.speaker_norm."),
    (r"^fusion\.proj\.", "model.multi_modal_projector."),
    (r"^ctc\.transformer\.", "model.word_timestamp_encoder."),
    (r"^ctc\.classifier\.", "word_timestamp_head."),
]
# Read by the checks, not loaded: the feature extractor and the speaker
# embedding buffer compute them.
CHECKED_ONLY = ("preprocessor.fb", "fusion.diar_kernel")


def hf_state_dict(tensors: dict) -> dict:
    """The reference model's tensors from the encoder GGUF's."""
    out = {}
    for name, a in tensors.items():
        if name in CHECKED_ONLY:
            continue
        if name == "ctc.upsample.weight":
            # conversion/ctc_timestamp.py stores the ConvTranspose1d weight
            # (in, out, U) as rows j * D + o, columns i.
            d = a.shape[1]
            u = a.shape[0] // d
            out["model.word_timestamp_encoder.upsampler.weight"] = (
                a.reshape(u, d, d).transpose(2, 1, 0))
            continue
        if name == "ctc.upsample.bias":
            d = tensors["ctc.upsample.weight"].shape[1]
            out["model.word_timestamp_encoder.upsampler.bias"] = a[:d]
            continue
        hf = name
        for pat, rep in _OUTER:
            if re.search(pat, hf):
                hf = re.sub(pat, rep, hf)
                break
        else:
            raise SystemExit(f"error: no reference name for {name}")
        for pat, rep in _INNER:
            hf = re.sub(pat, rep, hf)
        if hf.endswith(".attn.w_qkv.weight"):
            base = hf[: -len(".attn.w_qkv.weight")] + ".self_attn."
            q, k, v = np.split(a, 3, axis=0)
            out[base + "q_proj.weight"] = q
            out[base + "k_proj.weight"] = k
            out[base + "v_proj.weight"] = v
            continue
        if hf == "word_timestamp_head.weight":
            a = a[:, :, None]
        out[hf] = a
    return out


def reference_config(meta: dict):
    """The transformers config of the reference model, from the encoder
    GGUF's metadata."""
    from transformers import Nemotron3_5TranscribeConfig

    def branch(p):
        heads = int(meta[f"pee.{p}.n_heads"])
        d = int(meta[f"pee.{p}.d_model"])
        rot = int(meta[f"pee.{p}.rotary_dims"])
        return {
            "num_mel_bins": int(meta[f"pee.{p}.feat_in"]),
            "hidden_size": d,
            "intermediate_size": int(meta[f"pee.{p}.ff_hidden"]),
            "num_hidden_layers": int(meta[f"pee.{p}.n_layers"]),
            "num_attention_heads": heads,
            "subsampling_factor": int(meta[f"pee.{p}.subsampling_factor"]),
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": rot / (d // heads)},
        }

    asr = branch("asr")
    diar_audio = branch("diar")
    n_spk = int(meta["pee.diar.num_speakers"])
    diarization = {
        "audio_config": diar_audio,
        "head_config": {"hidden_size": int(meta["pee.diar.tf_d_model"]),
                        "num_speakers": n_spk,
                        "audio_hidden_size": diar_audio["hidden_size"],
                        "subsampling_factor": diar_audio["subsampling_factor"]},
        "streaming_config": {"speaker_cache_length": SPEAKER_CACHE_LENGTH,
                             "num_speakers": n_spk,
                             "subsampling_factor": diar_audio["subsampling_factor"],
                             **SPEAKER_CACHE_POLICY},
        "chunk_length": int(meta["pee.online.inference_length"]),
        "chunk_left_context": int(meta["pee.online.chunk_left_context"]),
        "chunk_right_context": int(meta["pee.online.chunk_right_context"]),
        "fifo_length": FIFO_LENGTH,
        "speaker_cache_update_period": SPEAKER_CACHE_UPDATE_PERIOD,
    }
    ctc_d = int(meta["ctc.d_model"])
    ctc_heads = int(meta["ctc.n_heads"])
    word = {
        "hidden_size": ctc_d,
        "intermediate_size": int(meta["ctc.ff_hidden"]),
        "num_hidden_layers": int(meta["ctc.n_layers"]),
        "num_attention_heads": ctc_heads,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor":
                                int(meta["ctc.rotary_dims"]) / (ctc_d // ctc_heads)},
        "vocab_size": int(meta["ctc.vocab_size"]),
        "blank_token_id": int(meta["ctc.vocab_size"]) - 1,
        "residual_scale": float(meta["ctc.residual_scale"]),
        "upsampling_factor": int(meta["ctc.upsampling_factor"]),
    }
    # A one-layer language model stand-in: only its width (the projector's
    # output) matters, and it never runs.
    text = {"hidden_size": int(meta["pee.fusion.proj_dim"]),
            "num_hidden_layers": 1, "hybrid_override_pattern": "*",
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "vocab_size": 16, "intermediate_size": 8,
            "mamba_num_heads": 2, "mamba_head_dim": 4, "ssm_state_size": 4,
            "n_groups": 1, "num_nextn_predict_layers": 0}
    return Nemotron3_5TranscribeConfig(
        audio_config={"asr_config": asr, "diarization_config": diarization,
                      "speaker_activity_threshold":
                          float(meta["pee.fusion.speaker_activity_threshold"])},
        text_config=text, word_timestamp_config=word)


def reference_model(meta: dict, tensors: dict, mode: str):
    """The transformers reference model with the encoder GGUF's weights, in
    ``mode``."""
    import torch
    from transformers import Nemotron3_5TranscribeForConditionalGeneration

    model = Nemotron3_5TranscribeForConditionalGeneration(reference_config(meta))
    sd = {k: torch.from_numpy(np.ascontiguousarray(v))
          for k, v in hf_state_dict(tensors).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"error: unexpected reference tensors {unexpected[:5]}")
    loaded = [k for k in missing
              if not k.startswith(("model.language_model.", "lm_head."))]
    if loaded:
        raise SystemExit(f"error: reference tensors left unloaded {loaded[:5]}")
    model.eval()
    return model.to(torch_dtype(mode))


def feature_extractor():
    from transformers import NemotronAsrStreamingFeatureExtractor

    return NemotronAsrStreamingFeatureExtractor(**FEATURES)


def features(audio: np.ndarray):
    """``(input_features, input_features_mask)`` of one 16 kHz clip, as the
    reference processor builds them (processing_nemotron3_5_transcribe.py
    ``_process_audio``): the extractor's spectrogram and mask, padded on
    the right to a multiple of ``SUBSAMPLING`` frames."""
    import torch

    out = feature_extractor()(audio, sampling_rate=16000, return_tensors="pt")
    feats = out["input_features"]
    mask = out["attention_mask"]
    pad = -feats.shape[1] % SUBSAMPLING
    feats = torch.nn.functional.pad(feats, (0, 0, 0, pad))
    mask = torch.nn.functional.pad(mask, (0, pad))
    return feats, mask


def load_audio(path: str | os.PathLike) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return audio.astype(np.float32)
