# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The media of a d1 decision: images, video frames and audio in the
state, and spoken questions on d1-Omni.

The request form is llama-server's (``server-decision.cpp``,
``parse_state``): top-level ``files`` and ``images`` hold data URLs of any
media, and a message-shaped state holds ``image_url``, ``input_audio`` and
``video_url`` parts, which are taken out of it. A video's frames are read,
not its soundtrack.

Each item is decoded and preprocessed once, on the request thread. Its
tower runs once, on the engine thread, when the first question that holds
it runs; every question's embeddings take the same features.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass, field
from typing import Any, Optional

from gmlx.systemone import d1

# The marker the template puts per media item: llama.cpp's media marker.
MARKER = "<__media__>"
MAX_MEDIA = 8
_KINDS = ("image", "audio", "video")
# Tower peaks measured on LFM3: 0.42 GB for a 2,048-token image, 0.39 GB
# for a video of any length, 80 MB plus 8 KB for each audio token.
_IMAGE_TOKEN_BYTES = 420_000_000 // 2048
_VIDEO_BYTES = 450_000_000
_AUDIO_BYTES = 80_000_000
_AUDIO_TOKEN_BYTES = 8_000


class D1MediaError(d1.D1RequestError):
    """A media item a d1 decision cannot take: a 400."""


@dataclass
class Item:
    """One media item: its kind, its processor inputs, the text its
    placeholder expands to, and its tower features once computed."""
    kind: str
    inputs: dict
    expansion: str
    features: Any = None


@dataclass
class Media:
    """The state with its media parts taken out, the state's items in
    order, and each spoken question's clip."""
    state: Any
    items: list = field(default_factory=list)
    spoken: dict = field(default_factory=dict)     # question id -> Item


def _data_url_kind(url) -> str:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise D1MediaError("media must be data URLs (data:image/..., data:audio/... "
                           "or data:video/...;base64,...)")
    mime = url[5:].split(";", 1)[0].split(",", 1)[0]
    kind = mime.split("/", 1)[0]
    if kind not in _KINDS:
        raise D1MediaError(f"media type {mime or 'unknown'} is not an image, audio "
                           "or video")
    return kind


def _audio_source(data) -> str:
    if not isinstance(data, str) or data.startswith(("http", "file://")):
        raise D1MediaError("audio must be base64 data")
    return data


def collect(body: dict) -> tuple[Any, list, dict]:
    """``(state, refs, spoken)``: the state with its media parts taken out,
    the state's media as ``(kind, source)`` in llama-server's order, and
    each spoken question's audio source."""
    if body.get("videos"):
        raise D1MediaError('"videos" is not supported: put a video in "files"')
    refs: list = []
    for key in ("files", "images"):
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise D1MediaError(f'"{key}" must be an array')
        for url in value:
            refs.append((_data_url_kind(url), url))
    state = body.get("state")
    wrapped = isinstance(state, dict) and "messages" in state
    messages = state["messages"] if wrapped else state
    if isinstance(messages, list):
        out = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                out.append(msg)
                continue
            kept = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind in ("image_url", "video_url") and kind in part:
                    url = part[kind]
                    url = url.get("url") if isinstance(url, dict) and "url" in url else url
                    refs.append((_data_url_kind(url), url))
                elif kind == "input_audio" and "input_audio" in part:
                    audio = part["input_audio"] if isinstance(part["input_audio"], dict) else {}
                    refs.append(("audio", _audio_source(audio.get("data", audio.get("url")))))
                else:
                    kept.append(part)
            out.append({**msg, "content": kept})
        if wrapped and isinstance(state, dict):
            state = dict(state, messages=out)
        else:
            state = out
    spoken = {}
    for qid, q in (body.get("questions") or {}).items():
        if isinstance(q, dict) and d1.is_spoken(q.get("instructions")):
            audio = q["instructions"].get("input_audio")
            audio = audio if isinstance(audio, dict) else {}
            spoken[qid] = _audio_source(audio.get("data"))
    if len(refs) + (1 if spoken else 0) > MAX_MEDIA:
        raise D1MediaError(f"too many media files, the maximum is {MAX_MEDIA}")
    return state, refs, spoken


def _audio_bytes(source: str) -> bytes:
    data = source.split(",", 1)[1] if source.startswith("data:") else source
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as e:
        raise D1MediaError(f"audio is not valid base64: {e}") from e


def _decode(kind: str, source: str, processor) -> Item:
    from gmlx._vendor.mlx_vlm import utils as vu
    from gmlx._vendor.mlx_vlm.models.base import to_mlx
    from gmlx.models.lfm3.processing import audio_expansion, image_expansion

    has_vision = getattr(processor, "image_processor", None) is not None
    if kind in ("image", "video") and not has_vision:
        raise D1MediaError(f"this model takes no {kind}s")
    if kind == "image":
        image = vu.load_image(source)
        pre = processor.image_processor.preprocess([image])
        grid = pre["image_grid_thw"][0]
        return Item(kind, to_mlx(dict(pre)),
                    image_expansion(processor.image_processor.soft_tokens(grid)))
    if kind == "video":
        vp = processor.video_processor
        sampling = vu.resolve_video_sampling(processor, {})
        frames, meta = vu.load_video(source, sampling, frame_sampler=vp.sample_frames)
        pre = vp.preprocess([frames])
        grid = pre["video_grid_thw"][0]
        return Item(kind, to_mlx(dict(pre)), processor._video_text(grid, meta, None))
    fe = getattr(processor, "feature_extractor", None)
    if fe is None:
        raise D1MediaError("this model takes no audio")
    samples = vu.load_audio(io.BytesIO(_audio_bytes(source)), sr=fe.sampling_rate)
    clip = processor._clip(samples)
    if fe.tokens(clip.size) == 0:
        raise D1MediaError(f"an audio clip of {clip.size} samples is too short")
    return Item(kind, to_mlx(dict(fe([clip]))),
                audio_expansion(fe.tokens(clip.size), has_vision))


def prepare(body: dict, processor) -> Media:
    """Decode and preprocess every media item of a request. Media gmlx
    cannot read is a :class:`D1MediaError`."""
    from gmlx.serve import media_decode
    from gmlx.serve.patches.media_gate import MediaRefused

    state, refs, spoken = collect(body)
    try:
        with media_decode.decode_pass():
            items = [_decode(kind, source, processor) for kind, source in refs]
            clips = {qid: _decode("audio", source, processor)
                     for qid, source in spoken.items()}
    except MediaRefused as e:
        raise D1MediaError(str(e)) from e
    except D1MediaError:
        raise
    except ValueError as e:
        raise D1MediaError(f"media could not be read: {e}") from e
    return Media(state, items, clips)


def expand(prompt: str, items: list) -> str:
    """Each media marker of a rendered prompt, in order, replaced by its
    item's expansion."""
    parts = prompt.split(MARKER)
    if len(parts) - 1 != len(items):
        raise RuntimeError(f"the prompt has {len(parts) - 1} media markers for "
                           f"{len(items)} items")
    out = [parts[0]]
    for item, text in zip(items, parts[1:]):
        out += [item.expansion, text]
    return "".join(out)


def build_passes(fmt: d1.D1Format, questions: list, media: Media, cap: int) -> list:
    """Each question's pass: its expanded prompt ids, its label groups, and
    the items its prompt holds, in order."""
    d1.check_spoken(fmt, questions)
    passes = []
    for q in questions:
        items = list(media.items)
        if q.id in media.spoken:
            items.append(media.spoken[q.id])
        prompt = fmt.prompt(media.state, q, n_media=len(media.items),
                            media_marker=MARKER)
        ids = fmt.encode(expand(prompt, items))
        if len(ids) > cap:
            raise d1.D1RequestError(
                f"questions.{q.id}: the prompt has {len(ids):,} tokens, media "
                f"included; a d1 prompt holds at most {cap:,}")
        passes.append(d1.Pass(q, ids, fmt.labels(q)[1], items=items))
    return passes


def _soft_tokens(item: Item) -> int:
    from gmlx.models.lfm3.processing import AUDIO_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN

    token = {"image": IMAGE_TOKEN, "video": VIDEO_TOKEN, "audio": AUDIO_TOKEN}[item.kind]
    return item.expansion.count(token)


def tower_bytes(item: Item) -> int:
    """The peak of one item's tower pass, beside the weights."""
    tokens = _soft_tokens(item)
    if item.kind == "image":
        return max(_IMAGE_TOKEN_BYTES * tokens, _IMAGE_TOKEN_BYTES * 1024)
    if item.kind == "video":
        return _VIDEO_BYTES
    return _AUDIO_BYTES + _AUDIO_TOKEN_BYTES * tokens


def media_bytes(media: Optional[Media], hidden: int) -> tuple[int, int]:
    """``(tower peak, features held)`` for a request's media: the largest
    single tower pass, and the features every item keeps until the
    request ends."""
    if media is None:
        return 0, 0
    items = list(media.items) + list(media.spoken.values())
    peak = max((tower_bytes(i) for i in items), default=0)
    held = sum(_soft_tokens(i) for i in items) * hidden * 2
    return peak, held


def _features(vlm, item: Item):
    """The item's tower features, computed once, on the engine thread."""
    import mlx.core as mx

    if item.features is None:
        x = item.inputs
        if item.kind == "image":
            feats = vlm.vision_features(x["pixel_values"], x["image_grid_thw"])
        elif item.kind == "video":
            feats = vlm.vision_features(x["pixel_values_videos"], x["video_grid_thw"])
        else:
            feats = vlm.audio_features(x["input_features"], x["feature_attention_mask"])
        mx.eval(feats)
        item.features = feats
    return item.features


def encoder(vlm):
    """The function that runs one item's tower and keeps its features."""
    def encode(item: Item) -> None:
        _features(vlm, item)

    return encode


def embedder(vlm):
    """The function that gives a media pass its input embeddings,
    ``[len(ids), hidden]``."""
    import mlx.core as mx

    from gmlx.models.lfm3.vlm_model import scatter_features

    token_ids = {"image": vlm.config.image_token_id,
                 "video": vlm.config.video_token_id,
                 "audio": vlm.config.audio_token_id}

    def embed(p: d1.Pass) -> mx.array:
        ids = mx.array([p.ids])
        embeds = vlm.language_model.model.embed_tokens(ids)
        for kind in _KINDS:
            feats = [_features(vlm, item) for item in p.items if item.kind == kind]
            if feats:
                joined = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)
                embeds = scatter_features(token_ids[kind], joined, embeds, ids, kind)
        return embeds[0]

    return embed


def media_model(model) -> Optional[Any]:
    """The LFM3 VLM that runs the towers, or None for a text-only load."""
    vlm = model
    if getattr(vlm, "language_model", None) is not None and (
            getattr(vlm, "vision_tower", None) is not None
            or getattr(vlm, "audio_encoder", None) is not None):
        return vlm
    return None
