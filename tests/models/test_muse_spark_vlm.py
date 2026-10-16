"""Muse Spark GGUF + mmproj: the audio encoder, the processor and the loader
rows for image and audio input.

The audio encoder is checked against a NumPy port of llama.cpp's
``clip_graph_muse_spark_audio::build`` on tiny random weights. CPU only, no
GGUF and no download.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import gmlx.load.vlm as gvlm  # noqa: E402
import gmlx.models.muse_glimmer.tools as muse_tools  # noqa: E402
import gmlx.models.muse_spark.vlm_model as vm  # noqa: E402
from gmlx.models.muse_spark.audio import AudioConfig, AudioModel  # noqa: E402

HIDDEN = 64
IMAGE_TOKEN, AUDIO_TOKEN, START, END = 60, 61, 62, 63
TINY_AUDIO = dict(hidden_sizes=[8, 12], patch_sizes=[4, 3], depths=[1, 2],
                  intermediate_sizes=[16, 20], conv_kernel_size=3,
                  rms_norm_eps=1e-6, projection_dim=HIDDEN)
PER = 12  # 4 * 3 samples per embedding
# 56x84 pixels: a 4x6 patch grid and 2x3 soft tokens after the 2x2 merge.
IMAGE_HW = (56, 84)
SOFT_TOKENS = 6


def _randomize(module, seed):
    rng = np.random.default_rng(seed)
    params = [(k, mx.array(rng.normal(0.0, 0.5, v.shape).astype(np.float32)))
              for k, v in tree_flatten(module.parameters())]
    module.update(tree_unflatten(params))
    mx.eval(module.parameters())


# NumPy port of llama.cpp's graph


def _np_rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _np_gelu(x):
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _np_audio(p, cfg: AudioConfig, wave):
    eps = cfg.rms_norm_eps
    x = wave.astype(np.float64).reshape(-1, 1)
    bid = 0
    for s, patch in enumerate(cfg.patch_sizes):
        T, C = x.shape
        x = x.reshape(T // patch, patch * C)
        pre = f"patches.{s}."
        x = x @ p[pre + "proj.weight"].T + p[pre + "proj.bias"]
        x = _np_rms(x, p[pre + "norm.weight"], eps)
        for _ in range(cfg.depths[s]):
            b = f"blocks.{bid}."
            w, K = p[b + "conv.weight"], cfg.conv_kernel_size
            conv = np.zeros_like(x)
            for t in range(x.shape[0]):
                for j in range(K):
                    src = t - (K - 1) + j
                    if src >= 0:
                        conv[t] += w[:, j] * x[src]
            conv = conv + p[b + "conv.bias"]
            h = _np_rms(conv, p[b + "conv_norm.weight"], eps)
            h = _np_gelu(h @ p[b + "mlp.fc1.weight"].T + p[b + "mlp.fc1.bias"])
            h = h @ p[b + "mlp.fc2.weight"].T + p[b + "mlp.fc2.bias"]
            x = x + h
            bid += 1
    x = _np_rms(x, p["norm.weight"], eps)
    x = x @ p["proj.weight"].T + p["proj.bias"]
    return _np_rms(x, p["post_norm.weight"], eps)


@pytest.fixture(scope="module")
def audio():
    cfg = AudioConfig(**TINY_AUDIO)
    model = AudioModel(cfg)
    _randomize(model, 1)
    params = {k: np.asarray(v, dtype=np.float64)
              for k, v in tree_flatten(model.parameters())}
    return model, cfg, params


def test_audio_encoder_matches_the_numpy_graph(audio):
    model, cfg, params = audio
    wave = np.random.default_rng(2).normal(0, 0.3, 5 * PER).astype(np.float32)
    got = np.asarray(model(mx.array(wave)))
    want = _np_audio(params, cfg, wave)
    assert got.shape == (5, HIDDEN)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_audio_embeddings_depend_only_on_earlier_samples(audio):
    """The trailing-placeholder rule for a reused prefix relies on this."""
    model, _, _ = audio
    wave = mx.array(np.random.default_rng(3).normal(0, 0.3, 6 * PER)
                    .astype(np.float32))
    full = model(wave)
    head = model(wave[: 4 * PER])
    assert mx.allclose(full[:4], head, atol=1e-5).item()


def test_audio_encoder_refuses_a_ragged_waveform(audio):
    model, _, _ = audio
    with pytest.raises(ValueError, match="pad the waveform"):
        model(mx.zeros((PER + 1,)))


# Loader rows


def _gguf_audio_arrays(params, cfg: AudioConfig):
    """The model's audio parameters under their mmproj names and layouts."""
    out = {"a.pre_ln.weight": params["norm.weight"],
           "mm.a.fc.weight": params["proj.weight"],
           "mm.a.fc.bias": params["proj.bias"],
           "a.post_ln.weight": params["post_norm.weight"]}
    for s in range(len(cfg.patch_sizes)):
        out[f"a.patch.{s}.weight"] = params[f"patches.{s}.proj.weight"]
        out[f"a.patch.{s}.bias"] = params[f"patches.{s}.proj.bias"]
        out[f"a.patch.{s}.norm.weight"] = params[f"patches.{s}.norm.weight"]
    for b in range(sum(cfg.depths)):
        w = params[f"blocks.{b}.conv.weight"]
        out[f"a.conv1d.{b}.weight"] = w.reshape(w.shape[0], 1, w.shape[1])
        out[f"a.conv1d.{b}.bias"] = params[f"blocks.{b}.conv.bias"]
        out[f"a.conv1d.{b}.norm.weight"] = params[f"blocks.{b}.conv_norm.weight"]
        for gguf, ours in (("ffn_up", "fc1"), ("ffn_down", "fc2")):
            for leaf in ("weight", "bias"):
                out[f"a.blk.{b}.{gguf}.{leaf}"] = params[
                    f"blocks.{b}.mlp.{ours}.{leaf}"]
    return {k: mx.array(v.astype(np.float32)) for k, v in out.items()}


def test_remap_fills_every_audio_parameter(audio):
    _, cfg, params = audio
    arrays = _gguf_audio_arrays(params, cfg)
    arrays["v.pre_ln.weight"] = mx.ones((4,))
    arrays["a.unknown.0.weight"] = mx.ones((4,))
    out, skipped, kq = gvlm.remap_vision_arrays(
        arrays, "muse_spark", with_audio=True)
    assert skipped == ["a.unknown.0.weight"] and kq == {}
    assert out.pop("vision_tower.pre_layernorm.weight").shape == (4,)
    want = {f"audio_tower.{k}": v.shape for k, v in params.items()}
    assert {k: tuple(v.shape) for k, v in out.items()} == want
    conv = out["audio_tower.blocks.0.conv.weight"]
    assert np.array_equal(np.asarray(conv), params["blocks.0.conv.weight"]
                          .astype(np.float32))
    # Without an audio encoder flag the audio rows are skipped.
    out, skipped, _ = gvlm.remap_vision_arrays(arrays, "muse_spark")
    assert not [k for k in out if k.startswith("audio_tower.")]
    assert len(skipped) == len(arrays) - 1


_SPARK_CLIP = {
    "clip.has_vision_encoder": True,
    "clip.vision.projector_type": "muse-glimmer",
    "clip.vision.projection_dim": 8192,
    "clip.vision.image_size": 896,
    "clip.vision.patch_size": 14,
    "clip.vision.embedding_length": 1536,
    "clip.vision.feed_forward_length": 8960,
    "clip.vision.block_count": 50,
    "clip.vision.attention.head_count": 16,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.spatial_merge_size": 2,
    "clip.has_audio_encoder": True,
    "clip.audio.projector_type": "muse-spark-audio",
    "clip.audio.projection_dim": 8192,
    "clip.audio.embedding_length": 2048,
    "clip.audio.feed_forward_length": 8192,
    "clip.audio.block_count": 8,
    "clip.audio.num_mel_bins": 1,
    "clip.audio.attention.layer_norm_epsilon": 1e-6,
    "clip.audio.conv_kernel_size": 7,
    "clip.audio.hidden_sizes": [256, 512, 1024, 2048],
    "clip.audio.patch_sizes": [16, 6, 5, 4],
    "clip.audio.depths": [2, 2, 2, 2],
}
# GGUF ne order, as the loader's shape map carries them.
_SPARK_SHAPES = {
    "v.position_embd.weight": (1536, 1024),
    "mm.0.weight": (6144, 4096),
    "a.blk.0.ffn_up.weight": (256, 1024),
    "a.blk.2.ffn_up.weight": (512, 2048),
    "a.blk.4.ffn_up.weight": (1024, 4096),
    "a.blk.6.ffn_up.weight": (2048, 8192),
}


def _spark_llm_meta():
    toks = [f"t{i}" for i in range(200100)]
    toks[200038], toks[200092] = "<|audio|>", "<|patch|>"
    return {"tokenizer.ggml.tokens": toks}


def test_config_reads_the_spark_mmproj_header():
    text = {"model_type": "muse_spark", "hidden_size": 8192,
            "vocab_size": 202048}
    cfg = gvlm._synthesize_muse_spark_vlm_config(
        text, _SPARK_CLIP, _spark_llm_meta(), _SPARK_SHAPES)
    assert cfg["model_type"] == "muse_spark"
    assert cfg["image_token_id"] == cfg["image_token_index"] == 200092
    assert cfg["audio_token_id"] == 200038
    v = cfg["vision_config"]
    assert (v["num_hidden_layers"], v["hidden_size"], v["patch_size"],
            v["projection_dim"], v["adapter_hidden_size"],
            v["num_position_embeddings"]) == (50, 1536, 14, 8192, 4096, 1024)
    a = cfg["audio_config"]
    assert a["hidden_sizes"] == [256, 512, 1024, 2048]
    assert a["patch_sizes"] == [16, 6, 5, 4]
    assert a["depths"] == [2, 2, 2, 2]
    assert a["intermediate_sizes"] == [1024, 2048, 4096, 8192]
    assert (a["conv_kernel_size"], a["projection_dim"], a["sample_rate"]) == (
        7, 8192, 24000)
    assert a["rms_norm_eps"] == pytest.approx(1e-6)
    assert AudioConfig.from_dict(a).samples_per_token == 1920


def test_config_refuses_a_projection_that_misses_the_text_width():
    text = {"model_type": "muse_spark", "hidden_size": 6656}
    with pytest.raises(ValueError, match="does not match"):
        gvlm._synthesize_muse_spark_vlm_config(
            text, _SPARK_CLIP, {}, _SPARK_SHAPES)


def test_config_takes_an_audio_only_mmproj():
    meta = {k: v for k, v in _SPARK_CLIP.items() if not k.startswith("clip.vision")}
    meta["clip.has_vision_encoder"] = False
    text = {"model_type": "muse_spark", "hidden_size": 8192}
    cfg = gvlm._synthesize_muse_spark_vlm_config(text, meta, {}, _SPARK_SHAPES)
    assert cfg["vision_config"] is None and cfg["audio_config"] is not None


def test_loader_keeps_the_vision_tower_at_its_file_dtype():
    from gmlx.load.loader import float_dtype_rules

    keep = float_dtype_rules("muse_spark")["f16_keep"]
    for prefix in ("vision_tower.", "vision_adapter.", "vision_projection."):
        assert prefix in keep
    # The audio weights are exact in bf16, so the audio tower takes the cast.
    assert not any(p.startswith("audio_tower") for p in keep)
    # The text tower's own fp32 rows still apply under the VLM type.
    assert ".mlp.gate.weight" in float_dtype_rules("muse_spark")["fp32_keep"]


# The VLM model


def _config(audio=True) -> dict:
    return {
        "model_type": "muse_spark",
        "text_config": {
            "model_type": "muse_spark",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "moe_latent_size": 32,
        },
        "vision_config": {
            "model_type": "muse_glimmer",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "image_size": 56,
            "patch_size": 14,
            "num_channels": 3,
            "projection_dim": HIDDEN,
            "adapter_hidden_size": 48,
            "spatial_merge_size": 2,
            "sparse_factor": 4,
            "num_position_embeddings": 16,
        },
        "audio_config": ({"model_type": "muse_spark_audio", **TINY_AUDIO}
                         if audio else None),
        "vocab_size": 64,
        "image_token_id": IMAGE_TOKEN,
        "image_token_index": IMAGE_TOKEN,
        "audio_token_id": AUDIO_TOKEN,
    }


@pytest.fixture(scope="module")
def model():
    muse_tools.ensure_registered()
    built, _ = gvlm.build_vlm_model(_config())
    _randomize(built, 4)
    return built


def test_loader_builds_the_gmlx_class(model):
    assert type(model) is vm.Model
    assert type(model.language_model) is vm.LanguageModel
    assert type(model.audio_tower) is AudioModel


def _waves(*counts):
    rng = np.random.default_rng(5)
    width = max(counts) * PER
    waves = np.zeros((len(counts), width), dtype=np.float32)
    mask = np.zeros((len(counts), max(counts)), dtype=bool)
    for i, n in enumerate(counts):
        waves[i, : n * PER] = rng.normal(0, 0.3, n * PER)
        mask[i, :n] = True
    return mx.array(waves), mx.array(mask)


def test_audio_features_land_on_the_audio_rows(model):
    waves, mask = _waves(3, 2)
    ids = mx.array([[1, START] + [AUDIO_TOKEN] * 3 + [END, 2, START]
                    + [AUDIO_TOKEN] * 2 + [END]])
    text = model.get_input_embeddings(ids).inputs_embeds
    merged = model.get_input_embeddings(
        ids, input_features=waves, input_features_mask=mask).inputs_embeds
    feats = mx.concatenate([model.audio_tower(waves[0, : 3 * PER]),
                            model.audio_tower(waves[1, : 2 * PER])])
    rows = [2, 3, 4, 8, 9]
    got = merged[0, mx.array(rows)]
    assert mx.allclose(got, feats.astype(got.dtype), atol=1e-5).item()
    others = [i for i in range(ids.shape[1]) if i not in rows]
    assert mx.array_equal(merged[0, mx.array(others)], text[0, mx.array(others)])


def test_a_reused_prefix_takes_the_last_audio_features(model):
    """Prompt-cache reuse trims input_ids to the suffix but keeps the
    waveform, so the remaining placeholders take the trailing features."""
    waves, mask = _waves(4)
    feats = model.audio_tower(waves[0])
    suffix = mx.array([[AUDIO_TOKEN, AUDIO_TOKEN, END, 2]])
    out = model.get_input_embeddings(
        suffix, input_features=waves, input_features_mask=mask).inputs_embeds
    assert mx.allclose(out[0, :2], feats[2:].astype(out.dtype), atol=1e-5).item()
    # A suffix past every placeholder is plain text.
    text = mx.array([[END, 2]])
    assert mx.array_equal(
        model.get_input_embeddings(text, input_features=waves).inputs_embeds,
        model.get_input_embeddings(text).inputs_embeds)


def test_more_placeholders_than_features_fail_by_name(model):
    waves, mask = _waves(1)
    ids = mx.array([[AUDIO_TOKEN] * 2])
    with pytest.raises(ValueError, match="2 audio placeholder tokens but 1"):
        model.get_input_embeddings(
            ids, input_features=waves, input_features_mask=mask)


def test_image_and_audio_in_one_prompt(model):
    waves, mask = _waves(2)
    ids = mx.array([[1] + [IMAGE_TOKEN] * SOFT_TOKENS + [START]
                    + [AUDIO_TOKEN] * 2 + [END, 3]])
    pixels = mx.random.normal((1, 3, *IMAGE_HW), key=mx.random.key(6))
    kwargs = dict(image_sizes=[IMAGE_HW], input_features=waves,
                  input_features_mask=mask)
    text = model.get_input_embeddings(ids).inputs_embeds
    merged = model.get_input_embeddings(ids, pixels, **kwargs).inputs_embeds
    assert not mx.allclose(merged[0, 1:7], text[0, 1:7]).item()
    assert not mx.allclose(merged[0, 8:10], text[0, 8:10]).item()
    out = model(ids, pixels, cache=model.make_cache(), **kwargs)
    assert out.logits.shape == (1, ids.shape[1], 64)


def test_a_model_without_an_audio_tower_refuses_audio():
    muse_tools.ensure_registered()
    built, _ = gvlm.build_vlm_model(_config(audio=False))
    assert not hasattr(built, "audio_tower")
    waves, mask = _waves(1)
    with pytest.raises(ValueError, match="audio encoder"):
        built.get_input_embeddings(mx.array([[AUDIO_TOKEN]]),
                                   input_features=waves)


# The processor

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
_SPECIALS = [BOS, "<|patch|>", "<|image_start|>", "<|image_end|>", EOS,
             "<|audio|>", "<|soft_audio_start|>", "<|soft_audio_end|>"]


def _processor():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1, "there": 2,
             **{t: 3 + i for i, t in enumerate(_SPECIALS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", special_tokens=[(BOS, vocab[BOS])])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=BOS, eos_token=EOS, unk_token="[UNK]")
    proc = gvlm._synthesize_vlm_processor("muse_spark", fast, _SPARK_CLIP)
    return proc, vocab


def _ids(feat):
    return [int(t) for t in feat["input_ids"][0].tolist()]


def test_processor_expands_each_clip_to_one_token_per_1920_samples():
    proc, v = _processor()
    fe = proc.feature_extractor
    assert (fe.sampling_rate, fe.samples_per_token) == (24000, 1920)
    assert [fe.tokens(n) for n in (1, 1920, 1921, 48000)] == [1, 1, 2, 25]
    clips = [np.ones(1921, np.float32), np.full(1920, 0.5, np.float32)]
    out = proc(text=f"{BOS}<|audio|> hi <|audio|>", audio=clips)
    a, s, e = v["<|audio|>"], v["<|soft_audio_start|>"], v["<|soft_audio_end|>"]
    assert _ids(out) == [v[BOS], s, a, a, e, 1, s, a, e]
    feats = np.asarray(out["input_features"])
    assert feats.shape == (2, 2 * 1920)
    assert feats[0, 1920] == 1.0 and feats[0, 1921:].sum() == 0
    assert feats[1, :1920].sum() == 960 and feats[1, 1920:].sum() == 0
    assert np.asarray(out["input_features_mask"]).tolist() == [
        [True, True], [True, False]]


def test_processor_takes_images_and_audio_together():
    from PIL import Image

    proc, v = _processor()
    image = Image.new("RGB", (IMAGE_HW[1], IMAGE_HW[0]))
    out = proc(images=[image], text=f"{BOS}<|patch|><|audio|> hi",
               audio=[np.zeros(100, np.float32)])
    p = v["<|patch|>"]
    assert _ids(out) == ([v[BOS], v["<|image_start|>"]] + [p] * SOFT_TOKENS
                         + [v["<|image_end|>"], v["<|soft_audio_start|>"],
                            v["<|audio|>"], v["<|soft_audio_end|>"], 1])
    assert "pixel_values" in out and "input_features" in out


def test_prepare_inputs_loads_audio_at_24_khz(monkeypatch):
    """mlx-vlm resamples to the processor's feature extractor rate. Spark's
    is 24 kHz, while models without one keep the 16 kHz default."""
    import gmlx._vendor.mlx_vlm.utils as vutils

    rates = []

    def load_audio(file, sr, timeout=10):
        rates.append(sr)
        return np.zeros(sr // 10, dtype=np.float32)

    monkeypatch.setattr(vutils, "load_audio", load_audio)
    proc, v = _processor()
    out = vutils.prepare_inputs(proc, audio=["clip.wav"],
                                prompts=f"{BOS}<|audio|> hi")
    assert rates == [24000]
    # 2400 samples is two embeddings.
    assert _ids(out).count(v["<|audio|>"]) == 2
    assert out["input_features"].shape == (1, 2 * 1920)
