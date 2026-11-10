"""EmbeddingGemma 2 from an MLX safetensors checkpoint.

A tiny checkpoint is written to a temporary folder from the oracle's seeded
weights: in the layout mlx-vlm saves (towers in MLX convolution layout) and
in transformers' (a ``model.`` prefix, PyTorch convolution layout). Each
loads to the vectors of the model the GGUF path builds from the same
weights, which the oracle tests check against transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.embedders import checkpoint as ckpt
from gmlx.models.embedders import gemma_embedding2 as eg2
from gmlx.models.embedders import gemma_embedding2_processor as proc

_ORACLE = Path(__file__).resolve().parents[1] / "fixtures" / "embeddinggemma2_oracle"


@pytest.fixture
def cpu(monkeypatch):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(eg2, "Q_BLOCK", 16)
    try:
        yield
    finally:
        mx.set_default_device(prev)


@pytest.fixture(scope="module")
def oracle():
    with np.load(_ORACLE / "text.npz") as t, np.load(_ORACLE / "media.npz") as m:
        return ({k: t[k] for k in t.files}, {k: m[k] for k in m.files})


def _config(media: dict, towers: bool = True) -> dict:
    config = dict(json.loads(str(media["config"])), model_type="embedding_gemma2")
    if not towers:
        config.pop("vision_config")
        config.pop("audio_config")
    return config


def _weights(oracle, layout: str) -> dict[str, mx.array]:
    text, media = oracle
    out = {k[2:]: mx.array(v) for k, v in text.items() if k.startswith("w.")}
    hf = {k[2:]: mx.array(v) for k, v in media.items() if k.startswith("w.")}
    if layout == "mlx":
        c = _config(media)
        hf = eg2.Towers(c["text_config"]["hidden_size"], c["vision_config"],
                        c["audio_config"]).sanitize(hf)
    out.update(hf)
    if layout == "transformers":
        out = {f"model.{k}": v for k, v in out.items()}
    return out


def _write(folder: Path, config: dict, weights: dict, *, pooling="mean",
           processor: dict | None = None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "",
         "type": "sentence_transformers.base.modules.transformer.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling",
         "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
    ]))
    (folder / "1_Pooling").mkdir(exist_ok=True)
    (folder / "1_Pooling" / "config.json").write_text(json.dumps(
        {"embedding_dimension": 48, "pooling_mode": pooling}))
    if processor is not None:
        (folder / "processor_config.json").write_text(json.dumps(processor))
    mx.save_safetensors(str(folder / "model.safetensors"), weights)
    return str(folder)


def _rel_l2(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _pixels(img: np.ndarray) -> mx.array:
    x = img.astype(np.float32) / 255.0
    x = x[None] if x.ndim == 3 else x
    return mx.array(x.transpose(0, 3, 1, 2))


def _mixed(model, m) -> np.ndarray:
    towers = model.towers
    image, video, audio = model.placeholder_ids
    media = {
        image: towers.encode_images(_pixels(m["image.image0"])),
        video: towers.encode_images(_pixels(m["video.frames"])),
        audio: towers.encode_audio(mx.array(m["audio.features1"][None]),
                                   mx.array(m["audio.mask1"][None])),
    }
    ids = mx.array(m["mixed.ids"][None])
    return np.array(model(ids, media=media).text_embeds[0].astype(mx.float32))


@pytest.mark.parametrize("layout", ["mlx", "transformers"])
def test_checkpoint_matches_the_gguf_model(tmp_path, oracle, cpu, layout):
    text, media = oracle
    path = _write(tmp_path / layout, _config(media), _weights(oracle, layout))
    model = ckpt.build(path)
    for case in ("short", "long"):
        vec = np.array(model(mx.array(text[f"{case}.ids"][None])).text_embeds[0])
        tol = max(10 * float(text[f"{case}.gap"][1]), 1e-5)
        assert _rel_l2(vec, text[f"{case}.vec"]) < tol, case
    assert ckpt.media_mismatch(path) is None
    model.towers = ckpt.load_towers(path)
    tol = max(10 * float(media["mixed.gap"][0]), 2e-5)
    assert _rel_l2(_mixed(model, media), media["mixed.vec"]) < tol


def test_bfloat16_checkpoint_keeps_its_dtype(tmp_path, oracle, cpu, monkeypatch):
    _, media = oracle
    weights = {k: v.astype(mx.bfloat16) for k, v in
               _weights(oracle, "mlx").items()}
    path = _write(tmp_path / "bf16", _config(media), weights)
    # Float32 activations, as on M1 and M2: the weights stay bfloat16.
    monkeypatch.setattr(ckpt, "activation_dtype", lambda mt: mx.float32)
    model = ckpt.build(path)
    model.towers = ckpt.load_towers(path)
    dtypes = {v.dtype for _, v in tree_flatten(model.parameters())}
    assert dtypes == {mx.bfloat16}
    out = model(mx.array(media["mixed.ids"][None][:, :8]))
    assert out.last_hidden_state.dtype == mx.float32
    # The same bfloat16 weights, loaded by hand under float32 activations.
    # (bfloat16 random towers are far from the float32 reference.)
    config = _config(media)
    want = eg2.Model(eg2.ModelArgs.from_dict(config))
    want.load_weights([(k, v) for k, v in weights.items()
                       if k.startswith("language_model.")], strict=True)
    want.language_model.act_dtype = mx.float32
    want.towers = eg2.Towers(config["text_config"]["hidden_size"],
                             config["vision_config"], config["audio_config"])
    want.towers.load_weights([(k, v) for k, v in weights.items()
                              if not k.startswith("language_model.")], strict=True)
    np.testing.assert_array_equal(_mixed(model, media), _mixed(want, media))


def test_float16_parameters_run_as_bfloat16(tmp_path, oracle, cpu, monkeypatch):
    _, media = oracle
    weights = {k: v.astype(mx.float16) for k, v in
               _weights(oracle, "mlx").items()}
    path = _write(tmp_path / "f16", _config(media, towers=False), weights)
    monkeypatch.setattr(ckpt, "activation_dtype", lambda mt: mx.bfloat16)
    model = ckpt.build(path)
    assert {v.dtype for _, v in tree_flatten(model.parameters())} == {mx.bfloat16}
    assert model.language_model.act_dtype is None


def test_quantized_checkpoint_loads_its_quantized_layers(tmp_path, oracle, cpu):
    text, media = oracle
    config = _config(media, towers=False)
    model = eg2.Model(eg2.ModelArgs.from_dict(config))
    model.load_weights([(k, v) for k, v in _weights(oracle, "mlx").items()
                        if k.startswith("language_model.")], strict=True)
    nn.quantize(model, group_size=32, bits=8,
                class_predicate=lambda _, m: hasattr(m, "to_quantized")
                and m.weight.shape[-1] % 32 == 0)
    config["quantization"] = {"group_size": 32, "bits": 8}
    path = _write(tmp_path / "q8", config, dict(tree_flatten(model.parameters())))
    got = ckpt.build(path)
    q = [n for n, m in got.named_modules() if isinstance(m, nn.QuantizedLinear)]
    assert "language_model.layers.0.self_attn.q_proj" in q
    assert "language_model.ple.per_layer_model_projection" in q
    # On the CPU device a quantized encoder runs float32.
    assert got.language_model.act_dtype == mx.float32
    vec = np.array(got(mx.array(text["long.ids"][None])).text_embeds[0])
    assert _rel_l2(vec, text["long.vec"]) < 2e-2
    assert ckpt.media_mismatch(path) == (
        "its config.json has no vision_config or audio_config")


def test_a_pooling_other_than_the_mean_is_refused(tmp_path, oracle):
    _, media = oracle
    path = _write(tmp_path / "cls", _config(media, towers=False),
                  _weights(oracle, "mlx"), pooling="cls")
    with pytest.raises(ckpt.UnsupportedEncoderError, match="pools the mean"):
        ckpt.build(path)


@pytest.mark.parametrize("value, mode", [
    ("mean", "mean"), ("cls", "cls"), ("lasttoken", "last"), ("max", None)])
def test_sentence_transformers_6_pooling_mode(tmp_path, value, mode):
    (tmp_path / "modules.json").write_text(json.dumps([
        {"idx": 1, "name": "1", "path": "1_Pooling",
         "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"}]))
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode": value}))
    if mode is None:
        with pytest.raises(ckpt.UnsupportedEncoderError, match="'max'"):
            ckpt.pooling_mode(str(tmp_path), "embedding_gemma2")
    else:
        assert ckpt.pooling_mode(str(tmp_path), "embedding_gemma2") == mode


def test_the_checkpoint_is_an_encoder_with_no_card_to_fetch(tmp_path, oracle):
    _, media = oracle
    folder = tmp_path / "bare"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(_config(media)))
    assert ckpt.check(str(folder)) == "embedding_gemma2"
    assert ckpt.card_files_to_fetch(str(folder)) == []
    assert "EmbeddingGemma 2" in ckpt.supported_models()


# The sections of google/embeddinggemma-2's processor_config.json that
# gmlx reads.
_PROCESSOR = {
    "feature_extractor": {
        "dither": 0.0, "feature_size": 128, "fft_length": 512,
        "frame_length": 320, "hop_length": 160, "input_scale_factor": 1.0,
        "max_frequency": 8000.0, "mel_floor": 0.001, "min_frequency": 0.0,
        "padding_side": "right", "preemphasis": 0.0, "sampling_rate": 16000},
    "image_processor": {
        "do_normalize": False, "do_rescale": True, "do_resize": True,
        "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3,
        "resample": 3, "rescale_factor": 0.00392156862745098},
    "video_processor": {
        "add_timestamps": False, "do_normalize": False, "do_rescale": True,
        "do_resize": True, "do_sample_frames": True, "fps": 1,
        "max_frames": 32, "max_soft_tokens": 140, "overflow_strategy": "uniform",
        "patch_size": 16, "pooling_kernel_size": 3, "resample": 3,
        "rescale_factor": 0.00392156862745098},
}


def test_the_model_processor_config_passes():
    assert proc.config_mismatch(_PROCESSOR) is None
    assert proc.config_mismatch({}) is None


@pytest.mark.parametrize("section, key, value, words", [
    ("video_processor", "fps", 2, "video_processor.fps to 2"),
    ("video_processor", "fps", 1.0, None),
    ("image_processor", "max_soft_tokens", 560, "max_soft_tokens to 560"),
    ("image_processor", "do_normalize", True, "do_normalize to True"),
    ("video_processor", "add_timestamps", 0, "add_timestamps to 0"),
    ("feature_extractor", "mel_floor", 1e-5, "mel_floor"),
])
def test_another_processor_config_is_named(section, key, value, words):
    config = json.loads(json.dumps(_PROCESSOR))
    config[section][key] = value
    got = proc.config_mismatch(config)
    assert got is None if words is None else words in got


def test_media_mismatch_reads_the_processor_config(tmp_path, oracle):
    _, media = oracle
    bad = json.loads(json.dumps(_PROCESSOR))
    bad["video_processor"]["max_frames"] = 64
    path = _write(tmp_path / "proc", _config(media), {"x": mx.zeros(1)},
                  processor=bad)
    assert "max_frames to 64" in ckpt.media_mismatch(path)


def test_prompts_come_from_the_sentence_transformers_config(tmp_path):
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps(
        {"prompts": {"query": "task: search result | query: ",
                     "document": "title: none | text: "}}))
    assert ckpt.prompts(str(tmp_path))["document"] == "title: none | text: "
    assert ckpt.prompts(str(tmp_path / "missing")) is None
