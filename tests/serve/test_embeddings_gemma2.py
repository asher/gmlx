"""EmbeddingGemma 2 behind POST /v1/embeddings, on the tiny oracle model
(gmlx.serve.embeddings with the model holder stubbed): text items, media
items decoded from the request, the refusals of media gmlx cannot take,
prompt_name, dimensions and the context limit. CPU only."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gmlx.serve.embeddings as emb
from gmlx.models.embedders import gemma_embedding2 as eg2
from gmlx.models.embedders import gemma_embedding2_processor as proc

_ORACLE = Path(__file__).resolve().parents[1] / "fixtures" / "embeddinggemma2_oracle"
GGUF = "/lib/embeddinggemma-2-Q8_0.gguf"


class _Tok:
    """Characters to ids 3..82, and the marker strings to the tiny model's
    marker ids unless they are split."""

    bos_token_id, eos_token_id = 2, 1
    SPECIAL = {"<|image|>": 90, "<|video|>": 91, "<|audio|>": 92,
               "<|image>": 93, "<image|>": 94, "<|audio>": 95, "<audio|>": 89}

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s, t in self.SPECIAL.items():
                if not split_special_tokens and text.startswith(s, i):
                    ids.append(t)
                    i += len(s)
                    break
            else:
                ids.append(3 + ord(text[i]) % 80)
                i += 1
        return ids


@pytest.fixture(scope="module")
def tiny():
    # Built on the service's worker, as the server loads its towers there:
    # MLX keeps lazy arrays on the stream of the thread that made them.
    def build():
        with np.load(_ORACLE / "text.npz") as t, np.load(_ORACLE / "media.npz") as m:
            config = json.loads(str(m["config"]))
            model = eg2.Model(eg2.ModelArgs.from_dict(config))
            model.load_weights([(k[2:], mx.array(v)) for k, v in t.items()
                                if k.startswith("w.")], strict=True)
            towers = eg2.Towers(config["text_config"]["hidden_size"],
                                config["vision_config"], config["audio_config"])
            weights = towers.sanitize({k[2:]: mx.array(v) for k, v in m.items()
                                       if k.startswith("w.")})
            towers.load_weights(list(weights.items()), strict=True)
            mx.eval(model.parameters(), towers.parameters())
        return model, towers

    return emb._EMBED_WORKER.submit(build).result()


@pytest.fixture
def served(tiny, monkeypatch):
    """The tiny model as the configured GGUF, with its towers as the
    projector's."""
    model, towers = tiny
    model.towers = None
    seen = []

    def get(cls, path):
        return model, _Tok()

    original = eg2.Model.__call__

    def forward(self, ids=None, *a, **kw):
        seen.append(np.array(ids)[0].tolist())
        return original(self, ids, *a, **kw)

    monkeypatch.setattr(emb._GGUFEmbeddingsHolder, "get", classmethod(get))
    monkeypatch.setattr(emb, "_load_towers", lambda m, t, s: towers)
    monkeypatch.setattr(emb._TowersHolder, "model_path", None)
    monkeypatch.setitem(emb._FAMILIES, GGUF, emb._Family(
        "EmbeddingGemma 2", "gemma2", emb._GEMMA2_PROMPTS, 768))
    monkeypatch.setattr(eg2.Model, "__call__", forward)
    return model, seen


def _run(body, media=emb.MediaSetting(path="/lib/mmproj-BF16.gguf")):
    return emb.run_embeddings(configured_model=GGUF, body=body, media=media)


def _png(w, h, color=(200, 30, 30)) -> str:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _wav(seconds, rate=16000) -> str:
    t = np.arange(int(seconds * rate)) / rate
    x = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return base64.b64encode(b.getvalue()).decode()


def test_text_items_read_placeholders_as_text(served):
    _, seen = served
    out = _run({"input": ["see <|image|> here", "plain"]})
    assert len(out["data"]) == 2 and len(out["data"][0]["embedding"]) == 48
    ids = seen[0]
    assert ids[0] == 2 and ids[-1] == 1                  # BOS ... EOS
    assert not {90, 91, 92} & set(ids)
    assert out["usage"]["prompt_tokens"] == sum(len(s) for s in seen)


def test_add_special_tokens_false_drops_bos_and_eos(served):
    _, seen = served
    _run({"input": "abc", "add_special_tokens": False})
    assert seen[-1] == _Tok().encode("abc")
    _run({"messages": [{"role": "user", "content": "abc"}],
          "add_special_tokens": False})
    assert seen[-1] == _Tok().encode("abc")


def test_prompt_name_comes_before_every_part(served):
    _, seen = served
    _run({"input": "abc", "prompt_name": "document"})
    assert seen[-1] == [2] + _Tok().encode("title: none | text: abc") + [1]


def test_an_image_item_runs_the_towers(served, tiny):
    model, seen = served
    url = _png(96, 48)
    out = _run({"input": [["a ", {"type": "image_url", "image_url": {"url": url}}]],
                "mm_processor_kwargs": {"max_soft_tokens": 70}})
    n = proc.vision_tokens(48, 96, 70)
    ids = seen[-1]
    assert ids.count(90) == n and ids.count(93) == 1 and ids.count(94) == 1
    assert out["usage"]["prompt_tokens"] == len(ids)
    # The same vector as the processor's own path.
    pixels = np.zeros((48, 96, 3), np.uint8)
    pixels[...] = (200, 30, 30)
    plan = proc.plan_item(_Tok(), ["a ", proc.Image(pixels)],
                          proc.Markers.from_args(model.args, 2, 1),
                          proc.Options(image_budget=70))
    model.towers = tiny[1]
    want = emb._EMBED_WORKER.submit(
        lambda: np.array(proc.encode_item(model, plan).text_embeds[0])).result()
    np.testing.assert_allclose(out["data"][0]["embedding"], want, atol=1e-5)


def test_an_audio_clip_is_cut(served, monkeypatch):
    # The tiny model holds 256 tokens, so the cut is 1 s here, not 30 s.
    _, seen = served
    monkeypatch.setattr(proc, "AUDIO_CUT", 16000)
    part = {"type": "input_audio", "input_audio": {"data": _wav(2.5), "format": "wav"}}
    _run({"input": [part]})
    assert seen[-1].count(92) == proc.audio_tokens(16000)


def test_a_lifted_cut_keeps_the_whole_clip(served, monkeypatch):
    _, seen = served
    monkeypatch.setattr(proc, "AUDIO_CUT", 16000)
    part = {"type": "input_audio", "input_audio": {"data": _wav(2.5), "format": "wav"}}
    _run({"input": [part], "mm_processor_kwargs": {"audio_kwargs": {"truncation": False}}})
    assert seen[-1].count(92) == proc.audio_tokens(40000)


def test_media_past_the_context_is_refused_before_the_worker(served, monkeypatch):
    _, seen = served
    worked = []
    monkeypatch.setattr(emb, "_embed_gemma2", lambda *a, **k: worked.append(1))
    url = _png(1000, 1000)
    images = [{"type": "image_url", "image_url": {"url": url}}] * 8
    with pytest.raises(emb.EmbeddingsRequestError, match="media tokens, over the 8192"):
        _run({"input": [images], "mm_processor_kwargs": {"max_soft_tokens": 1120}})
    assert not worked


@pytest.mark.parametrize("media, words", [
    (emb.MediaSetting(reason="server.embeddings_mmproj is false"),
     "runs text-only (server.embeddings_mmproj is false)"),
    (emb.MediaSetting(reason="no mmproj found; run gmlx pull hf:x/y/mmproj-BF16.gguf"),
     "(no mmproj found; run gmlx pull hf:x/y/mmproj-BF16.gguf)"),
    (emb.MediaSetting(), "no mmproj found; set server.embeddings_mmproj or "
                         "--embeddings-mmproj"),
])
def test_media_without_towers_is_refused(served, media, words):
    with pytest.raises(emb.EmbeddingsRequestError, match=words.replace("(", r"\(")
                       .replace(")", r"\)")) as ei:
        _run({"input": {"type": "image_url", "image_url": {"url": _png(48, 48)}}},
             media=media)
    assert ei.value.status_code == 400
    assert str(ei.value).startswith("media is not supported: the embeddings "
                                    "model runs text-only")


def test_a_projector_that_does_not_fit_is_named_once(served, monkeypatch):
    def bad(*a):
        raise ValueError("its vision projection_dim is 640, and the text "
                         "model's width is 512")

    monkeypatch.setattr(emb, "_load_towers", bad)
    body = {"input": {"type": "image_url", "image_url": {"url": _png(48, 48)}},
            "mm_processor_kwargs": {"max_soft_tokens": 70}}
    for _ in range(2):
        with pytest.raises(emb.EmbeddingsRequestError,
                           match="mmproj-BF16.gguf is not a projector for this "
                                 "model: its vision projection_dim is 640"):
            _run(body)
    # Text still embeds.
    assert len(_run({"input": "text"})["data"]) == 1


@pytest.mark.parametrize("family", [
    emb._Family("Qwen3-Embedding", "qwen3", emb._QWEN3_PROMPTS, 1024),
    emb._Family("EmbeddingGemma", "gemma", emb._GEMMA_PROMPTS, 768),
])
def test_other_models_take_text_only(monkeypatch, family):
    monkeypatch.setitem(emb._FAMILIES, GGUF, family)
    with pytest.raises(emb.EmbeddingsRequestError,
                       match=f"media is not supported: {family.label} takes "
                             "text only"):
        _run({"input": {"type": "image_url", "image_url": {"url": _png(48, 48)}}})
    with pytest.raises(emb.EmbeddingsRequestError,
                       match="chat messages are not supported"):
        _run({"messages": [{"role": "user", "content": "hi"}]})


def test_qwen3_prompt_name_and_dimensions(monkeypatch):
    monkeypatch.setitem(emb._FAMILIES, GGUF, emb._Family(
        "Qwen3-Embedding", "qwen3", emb._QWEN3_PROMPTS, 64))
    monkeypatch.setattr(emb._GGUFEmbeddingsHolder, "get",
                        classmethod(lambda cls, p: ("M", "T")))
    seen = {}

    def fake(model_obj, tok, texts, pooling="last", causal=True,
             add_special_tokens=True):
        seen["texts"] = texts
        rng = np.random.default_rng(0)
        m = rng.standard_normal((len(texts), 64)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True), 3

    monkeypatch.setattr(emb, "_embed_texts_gguf", fake)
    out = _run({"input": "what is a dog", "prompt_name": "query", "dimensions": 32})
    assert seen["texts"] == [emb._QWEN3_PROMPTS["query"] + "what is a dog"]
    vec = np.array(out["data"][0]["embedding"])
    assert vec.shape == (32,) and abs(np.linalg.norm(vec) - 1) < 1e-6
    _run({"input": "a dog", "prompt_name": "document"})
    assert seen["texts"] == ["a dog"]
    for dims in (16, 65):
        with pytest.raises(emb.EmbeddingsRequestError, match="from 32 to 64"):
            _run({"input": "x", "dimensions": dims})


def test_gemma_dimensions_cut_and_normalize_again(served):
    with pytest.raises(emb.EmbeddingsRequestError, match="768, 512, 256, 128"):
        _run({"input": "x", "dimensions": 100})
    m = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    np.testing.assert_allclose(emb._cut(m, 2), [[0.6, 0.8]])
    assert emb._cut(m, None) is m


def test_prompt_names_are_checked(served):
    with pytest.raises(emb.EmbeddingsRequestError, match="'nope' is not one of"):
        _run({"input": "x", "prompt_name": "nope"})


def test_a_model_without_a_table_refuses_prompt_name(monkeypatch):
    monkeypatch.setitem(emb._FAMILIES, GGUF, emb._Family("bert"))
    with pytest.raises(emb.EmbeddingsRequestError, match="bert has no prompt table"):
        _run({"input": "x", "prompt_name": "query"})


def test_text_cut_keeps_media_blocks():
    m = proc.Markers(bos=2, eos=1, boi=93, eoi=94, boa=95, eoa=89,
                     image=90, video=91, audio=92)
    ids = [2, 5, 6, 93, 90, 90, 94, 7, 8, 9, 1]
    cut = emb._cut_text(ids, 8, emb._protected(ids, m))
    assert cut == [2, 5, 6, 93, 90, 90, 94, 1]
