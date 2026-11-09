# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Google Inc. HuggingFace Inc. team. All rights reserved.
# Ported from transformers 5.22.2
# src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py and
# video_processing_embedding_gemma2.py, and modified for gmlx (Apache-2.0;
# see licenses/transformers-LICENSE).
"""Media preprocessing and token layout for EmbeddingGemma 2.

An item is a list of parts: text, images, audio clips and videos. Its
tokens are BOS, the parts in order, and EOS, as the model's chat template
and processor lay them out:

- text as it is, with no separator between parts;
- an image as ``<|image>``, its soft-token placeholders, ``<image|>``;
- a video as one such block per sampled frame, with ``<|video|>``
  placeholders and no timestamps;
- an audio clip as ``<|audio>``, its placeholders, ``<audio|>``.

The towers turn each image, frame and clip into as many soft tokens as it
has placeholders. The counts here are the processor's, from sizes alone, so
a request can be checked against the context before the towers run.

The values below are those of the model's ``processor_config.json``. A
GGUF carries none of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Soft tokens per image or frame that the vision tower supports.
VISION_BUDGETS = (70, 140, 280, 560, 1120)
IMAGE_BUDGET = 280
FRAME_BUDGET = 140
PATCH_SIZE = 16
POOLING_KERNEL = 3

VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 32

AUDIO_RATE = 16_000
# The reference cuts each clip at 30 s.
AUDIO_CUT = 480_000
_FRAME_LENGTH = 320          # 20 ms at 16 kHz
_HOP_LENGTH = 160            # 10 ms
_PAD_TO = 128                # the extractor pads clips to a multiple of this

IMAGE_TOKEN = "<|image|>"
AUDIO_TOKEN = "<|audio|>"
VIDEO_TOKEN = "<|video|>"
PLACEHOLDERS = (IMAGE_TOKEN, AUDIO_TOKEN, VIDEO_TOKEN)
BOI, EOI = "<|image>", "<image|>"
BOA, EOA = "<|audio>", "<audio|>"

# The processor_config.json values this module implements, by section. A
# checkpoint whose file sets another value runs text-only.
_CONFIG = {
    "image_processor": {
        "max_soft_tokens": IMAGE_BUDGET, "patch_size": PATCH_SIZE,
        "pooling_kernel_size": POOLING_KERNEL, "resample": 3,
        "do_resize": True, "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": False},
    "video_processor": {
        "max_soft_tokens": FRAME_BUDGET, "patch_size": PATCH_SIZE,
        "pooling_kernel_size": POOLING_KERNEL, "resample": 3,
        "do_resize": True, "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": False, "do_sample_frames": True, "fps": VIDEO_FPS,
        "max_frames": VIDEO_MAX_FRAMES, "overflow_strategy": "uniform",
        "add_timestamps": False},
    "feature_extractor": {
        "sampling_rate": AUDIO_RATE, "feature_size": 128,
        "frame_length": _FRAME_LENGTH, "hop_length": _HOP_LENGTH,
        "fft_length": 512, "mel_floor": 1e-3, "min_frequency": 0.0,
        "max_frequency": 8000.0, "preemphasis": 0.0, "dither": 0.0,
        "input_scale_factor": 1.0, "padding_side": "right"},
}


def config_mismatch(config: dict) -> str | None:
    """Why a checkpoint's ``processor_config.json`` (parsed as ``config``)
    asks for preprocessing this module does not run, or None. A value the
    file leaves out takes the module's own."""
    for section, want in _CONFIG.items():
        got = config.get(section) or {}
        if not isinstance(got, dict):
            return f"processor_config.json: {section} is not an object"
        for key, value in want.items():
            if key not in got:
                continue
            v = got[key]
            if isinstance(value, (bool, str)) or isinstance(v, bool):
                same = type(v) is type(value) and v == value
            else:
                same = (isinstance(v, (int, float))
                        and math.isclose(v, value, rel_tol=1e-9))
            if not same:
                return (f"processor_config.json sets {section}.{key} to "
                        f"{v!r}; gmlx runs {value!r}")
    return None


@dataclass(frozen=True)
class Options:
    """Per-request processor options (vLLM's ``mm_processor_kwargs``)."""

    image_budget: int = IMAGE_BUDGET
    frame_budget: int = FRAME_BUDGET
    # False lifts the 30 s audio cut.
    audio_cut: bool = True


def parse_options(kwargs: dict | None) -> Options:
    """:class:`Options` from a request's ``mm_processor_kwargs``. A flat
    ``max_soft_tokens`` sets both vision budgets, as transformers routes a
    kwarg that both processors take; ``images_kwargs`` and
    ``videos_kwargs`` set one each and win over the flat value. Raises
    ``ValueError`` on any other key or value."""
    if kwargs is None:
        return Options()
    if not isinstance(kwargs, dict):
        raise ValueError("mm_processor_kwargs must be an object")
    known = ("max_soft_tokens", "images_kwargs", "videos_kwargs", "audio_kwargs")
    extra = sorted(set(kwargs) - set(known))
    if extra:
        raise ValueError(f"unknown mm_processor_kwargs key {extra[0]!r}; "
                         f"known: {', '.join(known)}")

    def section(name, allowed):
        sub = kwargs.get(name, {})
        if not isinstance(sub, dict):
            raise ValueError(f"mm_processor_kwargs.{name} must be an object")
        extra = sorted(set(sub) - set(allowed))
        if extra:
            raise ValueError(f"mm_processor_kwargs.{name} takes "
                             f"{', '.join(allowed)}; got {', '.join(extra)}")
        return sub

    def budget(where, value):
        if (isinstance(value, bool) or not isinstance(value, int)
                or value not in VISION_BUDGETS):
            raise ValueError(
                f"{where} must be one of "
                f"{', '.join(map(str, VISION_BUDGETS))}, got {value!r}")
        return value

    flat = (budget("max_soft_tokens", kwargs["max_soft_tokens"])
            if "max_soft_tokens" in kwargs else None)
    images = section("images_kwargs", ("max_soft_tokens",))
    videos = section("videos_kwargs", ("max_soft_tokens",))
    audio = section("audio_kwargs", ("truncation",))
    image = (budget("images_kwargs.max_soft_tokens", images["max_soft_tokens"])
             if "max_soft_tokens" in images else flat)
    frame = (budget("videos_kwargs.max_soft_tokens", videos["max_soft_tokens"])
             if "max_soft_tokens" in videos else flat)
    cut = audio.get("truncation", True)
    if not isinstance(cut, bool):
        raise ValueError("audio_kwargs.truncation must be true or false")
    return Options(image_budget=image or IMAGE_BUDGET,
                   frame_budget=frame or FRAME_BUDGET, audio_cut=cut)


# Vision

def vision_size(height: int, width: int, budget: int) -> tuple[int, int]:
    """transformers' ``get_aspect_ratio_preserving_size``: the largest
    size of the image's aspect ratio with at most ``budget`` soft tokens,
    each side a multiple of 48 pixels."""
    max_patches = budget * POOLING_KERNEL ** 2
    target_px = max_patches * PATCH_SIZE ** 2
    factor = math.sqrt(target_px / (height * width))
    side = POOLING_KERNEL * PATCH_SIZE
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    if th == 0 and tw == 0:
        raise ValueError(f"a {width}x{height} image resizes to 0x0 pixels")
    max_side = (max_patches // POOLING_KERNEL ** 2) * side
    if th == 0:
        th = side
        tw = min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw = side
        th = min(int(math.floor(height / width)) * side, max_side)
    if th * tw > target_px:
        raise ValueError(
            f"a {width}x{height} image resizes to {tw}x{th} pixels, past "
            f"{budget} soft tokens")
    return th, tw


def vision_tokens(height: int, width: int, budget: int) -> int:
    """Soft tokens of one image or frame of ``height`` x ``width``."""
    th, tw = vision_size(height, width, budget)
    return (th // PATCH_SIZE) * (tw // PATCH_SIZE) // POOLING_KERNEL ** 2


def resize(frames: np.ndarray, budget: int) -> np.ndarray:
    """``[N, H, W, 3]`` uint8 images of one size, resized for ``budget`` as
    ``[N, 3, H', W']`` float32 in [0, 1]. Bicubic, as the reference; PIL's
    filter and torchvision's differ slightly."""
    from PIL import Image

    n, h, w = frames.shape[:3]
    th, tw = vision_size(h, w, budget)
    out = np.empty((n, 3, th, tw), dtype=np.float32)
    for i in range(n):
        img = frames[i]
        if (th, tw) != (h, w):
            img = np.asarray(Image.fromarray(img).resize(
                (tw, th), resample=Image.BICUBIC))
        out[i] = img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    return out


def sample_frames(total: int, fps: float | None, duration: float | None,
                  target_fps: float = VIDEO_FPS,
                  max_frames: int = VIDEO_MAX_FRAMES,
                  overflow: str = "uniform") -> np.ndarray:
    """``EmbeddingGemma2VideoProcessor.sample_frames``: the frame indices of
    a ``total``-frame video at ``target_fps``, capped at ``max_frames``.
    Without the video's fps or duration, every frame, capped."""
    if not fps or duration is None:
        indices = np.arange(total, dtype=int)
    else:
        step = fps / target_fps
        n = max(1, int(duration * target_fps))
        indices = np.array([min(total - 1, int(i * step)) for i in range(n)],
                           dtype=int)
    if len(indices) > max_frames:
        if overflow == "truncate":
            indices = indices[:max_frames]
        elif overflow == "uniform":
            pick = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = indices[pick]
        else:
            raise ValueError(f"unknown overflow strategy {overflow!r}")
    return indices


def frame_sampler(source, fps=None, max_frames=None):
    """``media_decode.read_video``'s ``frame_sampler``: the reference's
    sampling over the video's metadata."""
    return sample_frames(source.total_num_frames, source.fps, source.duration,
                         target_fps=fps or VIDEO_FPS,
                         max_frames=max_frames or VIDEO_MAX_FRAMES)


# Audio

def audio_mask(n_samples: int) -> np.ndarray:
    """The extractor's feature mask of a clip of ``n_samples`` (True at
    real frames): it pads the clip to a multiple of 128 samples, pads 160
    zeros in front, and takes 321-sample windows every 160 samples. A frame
    is real when the last sample of its window is."""
    if n_samples <= 0:
        return np.zeros(0, dtype=bool)
    padded = -(-n_samples // _PAD_TO) * _PAD_TO
    pad_left = _FRAME_LENGTH // 2
    real = np.zeros(padded + pad_left, dtype=bool)
    real[pad_left:pad_left + n_samples] = True
    size = _FRAME_LENGTH + 1
    n_frames = (len(real) - size) // _HOP_LENGTH + 1
    if n_frames <= 0:
        return np.zeros(0, dtype=bool)
    return real[np.arange(n_frames) * _HOP_LENGTH + size - 1]


def audio_tokens_from_mask(mask: np.ndarray) -> int:
    """Soft tokens of a clip with feature mask ``mask``: its real frames
    after the audio tower's two stride-2 subsamplings."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(2):
        t = len(m)
        m = m[::2][:(t + 2 - 3) // 2 + 1]
    return int(m.sum())


def audio_tokens(n_samples: int) -> int:
    """Soft tokens of a 16 kHz clip of ``n_samples``, after any cut."""
    return audio_tokens_from_mask(audio_mask(n_samples))


def cut_audio(samples: np.ndarray, options: Options) -> np.ndarray:
    """``samples`` cut at 30 s unless ``options`` lifts the cut. The cut
    is a copy, so the full decode can be freed."""
    if options.audio_cut and len(samples) > AUDIO_CUT:
        return np.array(samples[:AUDIO_CUT])
    return samples


_EXTRACTOR = None


def audio_features(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``([1, frames, 128], [1, frames])`` log-mel features and mask of one
    mono 16 kHz clip, already cut."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from gmlx._vendor.mlx_vlm.models.gemma4.audio_feature_extractor import (
            Gemma4AudioFeatureExtractor,
        )
        import warnings

        with warnings.catch_warnings():
            # The 128-bin mel bank at a 512-point FFT leaves a filter empty,
            # as the reference extractor does.
            warnings.filterwarnings("ignore", message="At least one mel filter")
            _EXTRACTOR = Gemma4AudioFeatureExtractor()
    out = _EXTRACTOR(np.asarray(samples, dtype=np.float32), max_length=None,
                     truncation=False)
    return out["input_features"][0][None], out["input_features_mask"][0][None]


# Token layout

def image_block(n: int) -> str:
    return BOI + IMAGE_TOKEN * n + EOI


def video_block(n_frames: int, per_frame: int) -> str:
    return (BOI + VIDEO_TOKEN * per_frame + EOI) * n_frames


def audio_block(n: int) -> str:
    return BOA + AUDIO_TOKEN * n + EOA


def encode_text(tokenizer, text: str) -> list[int]:
    """``text`` as token ids with the three placeholders read as plain
    text: an item without media must not hold placeholder ids. The begin
    and end markers keep their ids."""
    ids: list[int] = []
    rest = text
    while rest:
        hits = [(rest.find(p), p) for p in PLACEHOLDERS if p in rest]
        if not hits:
            ids += tokenizer.encode(rest, add_special_tokens=False)
            break
        at, p = min(hits)
        if at:
            ids += tokenizer.encode(rest[:at], add_special_tokens=False)
        ids += tokenizer.encode(p, add_special_tokens=False,
                                split_special_tokens=True)
        rest = rest[at + len(p):]
    return ids


@dataclass(frozen=True)
class Markers:
    """The marker token ids, from the model config."""

    bos: int
    eos: int
    boi: int
    eoi: int
    boa: int
    eoa: int
    image: int
    video: int
    audio: int

    @classmethod
    def from_args(cls, args, bos: int, eos: int) -> "Markers":
        return cls(bos=bos, eos=eos, boi=args.boi_token_id,
                   eoi=args.eoi_token_id, boa=args.boa_token_id,
                   eoa=args.eoa_token_index, image=args.image_token_id,
                   video=args.video_token_id, audio=args.audio_token_id)


@dataclass(frozen=True)
class Media:
    """One media part of an item, by its soft-token count: an image
    (``frames`` 1), a video of ``frames`` frames, or an audio clip."""

    kind: str          # "image", "video" or "audio"
    tokens: int        # per image, per frame, or per clip
    frames: int = 1

    @property
    def total(self) -> int:
        return self.tokens * self.frames


_PLACEHOLDER_OF = {"image": IMAGE_TOKEN, "video": VIDEO_TOKEN,
                   "audio": AUDIO_TOKEN}


def layout(tokenizer, parts: list, markers: Markers, *, prompt: str = "",
           add_special_tokens: bool = True) -> list[int]:
    """Token ids of an item: ``parts`` is a list of text strings and
    :class:`Media`, in order; ``prompt`` is the task prefix, which comes
    before all parts.

    As in the model's chat template, text that names placeholders in an
    item with media takes the media in their place, in order, and the media
    parts add no placeholders of their own. Raises ``ValueError`` when the
    placeholders and the media do not pair up. In an item without media a
    placeholder is plain text."""
    media = [p for p in parts if isinstance(p, Media)]
    texts = [p for p in parts if isinstance(p, str)]
    manual = bool(media) and any(ph in t for t in texts for ph in PLACEHOLDERS)

    def block(m: Media) -> list[int]:
        if m.kind == "audio":
            return [markers.boa] + [markers.audio] * m.tokens + [markers.eoa]
        tok = markers.image if m.kind == "image" else markers.video
        return ([markers.boi] + [tok] * m.tokens + [markers.eoi]) * m.frames

    # Runs of text and media blocks, with adjacent text joined.
    runs: list = [prompt] if prompt else []
    if manual:
        queues = {k: [m for m in media if m.kind == k] for k in _PLACEHOLDER_OF}
        text = "".join(texts)
        while text:
            hits = [(text.find(ph), k, ph) for k, ph in _PLACEHOLDER_OF.items()
                    if ph in text]
            if not hits:
                runs.append(text)
                break
            at, kind, ph = min(hits)
            runs.append(text[:at])
            if not queues[kind]:
                raise ValueError(
                    f"the text names more {ph} placeholders than the item "
                    f"has {kind} parts")
            runs.append(queues[kind].pop(0))
            text = text[at + len(ph):]
        left = [k for k, q in queues.items() if q]
        if left:
            raise ValueError(
                f"the item has more {left[0]} parts than its text names "
                f"{_PLACEHOLDER_OF[left[0]]} placeholders")
    else:
        runs += parts
    ids: list[int] = [markers.bos] if add_special_tokens else []
    pending = ""
    for run in runs + [None]:
        if isinstance(run, str):
            pending += run
            continue
        if pending:
            ids += (tokenizer.encode(pending, add_special_tokens=False)
                    if media else encode_text(tokenizer, pending))
            pending = ""
        if isinstance(run, Media):
            ids += block(run)
    if add_special_tokens:
        ids.append(markers.eos)
    return ids


# Items

@dataclass
class Image:
    """A decoded image, ``[H, W, 3]`` uint8."""

    pixels: np.ndarray


@dataclass
class Video:
    """A decoded video's sampled frames, ``[T, H, W, 3]`` uint8."""

    frames: np.ndarray


@dataclass
class Audio:
    """A decoded clip, mono float32 at 16 kHz, already cut."""

    samples: np.ndarray


def media_of(part, options: Options) -> Media:
    """The soft-token counts of a decoded part."""
    if isinstance(part, Image):
        h, w = part.pixels.shape[:2]
        return Media("image", vision_tokens(h, w, options.image_budget))
    if isinstance(part, Video):
        t, h, w = part.frames.shape[:3]
        return Media("video", vision_tokens(h, w, options.frame_budget), t)
    if isinstance(part, Audio):
        return Media("audio", audio_tokens(len(part.samples)))
    raise TypeError(f"not a media part: {type(part).__name__}")


@dataclass
class Plan:
    """An item ready for the encoder: its token ids and its decoded media
    in order."""

    ids: list
    media: list
    options: Options

    @property
    def tokens(self) -> int:
        return len(self.ids)


def plan_item(tokenizer, parts: list, markers: Markers, options: Options, *,
              prompt: str = "", add_special_tokens: bool = True) -> Plan:
    """The :class:`Plan` of an item of text strings and decoded media.
    Counts only; no pixels are resized."""
    laid = [p if isinstance(p, str) else media_of(p, options) for p in parts]
    ids = layout(tokenizer, laid, markers, prompt=prompt,
                 add_special_tokens=add_special_tokens)
    return Plan(ids=ids, media=[p for p in parts if not isinstance(p, str)],
                options=options)


# Frames per vision tower call: the frames of a video run in chunks, so the
# tower's transients stay those of a few frames.
FRAME_CHUNK = 4


def encode_item(model, plan: Plan, frame_chunk: int = FRAME_CHUNK):
    """The model's output for ``plan``: each image and chunk of frames
    through the vision tower, each clip through the audio tower alone, the
    soft tokens merged in order."""
    import mlx.core as mx

    towers = model.towers
    image_id, video_id, audio_id = model.placeholder_ids
    feats: dict = {image_id: [], video_id: [], audio_id: []}
    for part in plan.media:
        if isinstance(part, Image):
            px = resize(part.pixels[None], plan.options.image_budget)
            out = towers.encode_images(mx.array(px))
            mx.eval(out)
            feats[image_id].append(out)
        elif isinstance(part, Video):
            for i in range(0, len(part.frames), frame_chunk):
                px = resize(part.frames[i:i + frame_chunk],
                            plan.options.frame_budget)
                out = towers.encode_images(mx.array(px))
                mx.eval(out)
                feats[video_id].append(out)
        else:
            f, m = audio_features(part.samples)
            out = towers.encode_audio(mx.array(f), mx.array(m))
            mx.eval(out)
            feats[audio_id].append(out)
    media = {k: mx.concatenate(v) for k, v in feats.items() if v}
    return model(mx.array([plan.ids]), media=media or None)
