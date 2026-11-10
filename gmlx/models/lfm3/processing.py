# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Image and video preprocessing for LFM3-VL and LFM3-Omni.

Ported from Liquid AI's ``image_processing_lfm3_vl.py``,
``video_processing_lfm3_vl.py``, ``processing_lfm3_vl.py`` and
``processing_lfm3_omni.py`` (Apache-2.0):

- an image is resized to a multiple of ``patch * merge`` on each side with
  its pixel count inside the bounds, then cut into patches. Each patch is
  flattened ``(channel, temporal, y, x)`` and the patches are grouped by
  merge block. A still image fills both temporal slices;
- a video is sampled at 2 frames per second, between ``min_frames`` and
  ``max_frames``. All its frames share one size, set by the pixel budget
  of the whole clip, and each pair of frames is one temporal slice;
- in the prompt, an image becomes its soft tokens between the vision
  markers. A video becomes one group per frame pair inside one pair of
  markers. Each group is a ``<t seconds>`` label, then the pair's soft
  tokens between markers. ``t`` is the mean time of the pair's frames.

The chat template writes each placeholder between the vision markers, and
the expansion adds its own pair. The template's pair is removed first, as
the reference does.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from transformers.image_processing_utils import ImageProcessingMixin

IMAGE_TOKEN = "<|image|>"
VIDEO_TOKEN = "<|video|>"
AUDIO_TOKEN = "<|audio|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"

# llama.cpp caps an image at this many tokens (clip.cpp, lfm3vl).
IMAGE_TOKEN_CAP = 4096
# The reference video processor's pixel bounds for a whole clip, and its
# frame limit per model type. The mmproj has no key for them.
VIDEO_MIN_PIXELS = 4096
VIDEO_MAX_PIXELS = 25_165_824
MAX_FRAMES = {"lfm3_omni": 384, "lfm3_vl": 768}


def smart_resize(height: int, width: int, factor: int, min_pixels: int,
                 max_pixels: int) -> Tuple[int, int]:
    """The size an image is resized to: both sides a multiple of
    ``factor``, the pixel count within the bounds, the aspect ratio kept
    as closely as that allows."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def video_smart_resize(num_frames: int, height: int, width: int,
                       temporal_factor: int, factor: int, min_pixels: int,
                       max_pixels: int) -> Tuple[int, int]:
    """The size every frame of a clip is resized to. The pixel bounds
    count the whole clip: frames times height times width."""
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than "
            f"temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(frames: np.ndarray, patch_size: int, merge_size: int,
             temporal_patch_size: int):
    """``frames`` is ``[T, C, H, W]``. Returns the flattened patches,
    ``[gt * gh * gw, C * temporal * patch * patch]``, and ``(gt, gh, gw)``.
    A frame count that is not a multiple of the temporal patch repeats
    the last frame, so a still image fills every slice."""
    count, channel, height, width = frames.shape
    pad = -count % temporal_patch_size
    if pad:
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
        count += pad
    grid_t = count // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = frames.reshape(
        grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size)
    # [t, h/m, w/m, m, m, channel, temporal, patch, patch]
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size)
    return np.ascontiguousarray(flat), (grid_t, grid_h, grid_w)


def sample_frame_indices(total_frames: int, source_fps: float, fps: float,
                         min_frames: int, max_frames: int) -> np.ndarray:
    """The frames the reference video processor reads from a clip."""
    if not (math.isfinite(source_fps) and source_fps > 0):
        raise ValueError(
            f"the video's fps must be a positive number, got {source_fps}")
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a positive number, got {fps}")
    count = int(total_frames / source_fps * fps)
    count = min(max(count, min_frames), max_frames, total_frames)
    return np.linspace(0, total_frames - 1, count).round().astype(int)


def frame_timestamps(indices: Sequence[int], fps: float,
                     temporal_patch_size: int) -> list:
    """Seconds into the clip of each temporal slice: the mean time of its
    first and last frame."""
    indices = [int(i) for i in indices]
    pad = -len(indices) % temporal_patch_size
    if pad:
        indices.extend([indices[-1]] * pad)
    return [
        (indices[i] + indices[i + temporal_patch_size - 1]) / (2 * fps)
        for i in range(0, len(indices), temporal_patch_size)
    ]


def _resize_rgb(image, height: int, width: int) -> np.ndarray:
    """A PIL image as ``[3, height, width]`` uint8, bicubic."""
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


class _Geometry(ImageProcessingMixin):
    def __init__(self, patch_size: int, merge_size: int,
                 temporal_patch_size: int, min_pixels: int, max_pixels: int,
                 image_mean, image_std):
        super().__init__()
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.image_mean = [float(v) for v in image_mean]
        self.image_std = [float(v) for v in image_std]

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        """``[T, 3, H, W]`` uint8 to float32, rescaled and normalized."""
        mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
        return (frames.astype(np.float32) * np.float32(1.0 / 255.0)
                - mean) / std

    def soft_tokens(self, grid) -> int:
        """Text tokens of one temporal slice of a ``(t, h, w)`` grid."""
        return int(grid[1]) * int(grid[2]) // self.merge_size ** 2


class Lfm3ImageProcessor(_Geometry):
    """Resizes and patchifies still images. Returns ``pixel_values``, the
    patches of every image in order, and ``image_grid_thw``."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def target_size(self, height: int, width: int) -> Tuple[int, int]:
        return smart_resize(height, width, self.factor, self.min_pixels,
                            self.max_pixels)

    def preprocess(self, images, **kwargs) -> dict:
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        patches, grids = [], []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            width, height = image.size
            h_bar, w_bar = self.target_size(height, width)
            frame = self._normalize(_resize_rgb(image, h_bar, w_bar)[None])
            flat, grid = patchify(frame, self.patch_size, self.merge_size,
                                  self.temporal_patch_size)
            patches.append(flat)
            grids.append(grid)
        return {
            "pixel_values": (patches[0] if len(patches) == 1
                             else np.concatenate(patches, axis=0)),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }

    __call__ = preprocess


class Lfm3VideoProcessor(_Geometry):
    """Resizes and patchifies sampled clips. Returns
    ``pixel_values_videos``, the patches of every clip in order, and
    ``video_grid_thw``. The loader asks :meth:`sample_frames` which frames
    to read."""

    model_input_names = ["pixel_values_videos", "video_grid_thw"]
    sample_frames_in_loader = True

    def __init__(self, *args, fps: float = 2.0, min_frames: int = 4,
                 max_frames: int = 768, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = float(fps)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)

    def sample_frames(self, metadata, fps=None, max_frames=None, **kwargs):
        return sample_frame_indices(
            int(metadata.total_num_frames), float(metadata.fps),
            self.fps if fps is None else float(fps), self.min_frames,
            self.max_frames if max_frames is None else int(max_frames))

    def target_size(self, num_frames: int, height: int,
                    width: int) -> Tuple[int, int]:
        return video_smart_resize(
            num_frames, height, width, self.temporal_patch_size, self.factor,
            self.min_pixels, self.max_pixels)

    def preprocess(self, videos, **kwargs) -> dict:
        from PIL import Image

        patches, grids = [], []
        for clip in videos:
            clip = np.asarray(clip)
            if clip.ndim != 4:
                raise ValueError(
                    "a video is an array of frames, [T, C, H, W], got shape "
                    f"{clip.shape}")
            count, _, height, width = clip.shape
            h_bar, w_bar = self.target_size(count, height, width)
            frames = np.stack([
                _resize_rgb(Image.fromarray(
                    np.ascontiguousarray(frame.transpose(1, 2, 0))),
                    h_bar, w_bar)
                for frame in clip])
            flat, grid = patchify(self._normalize(frames), self.patch_size,
                                  self.merge_size, self.temporal_patch_size)
            patches.append(flat)
            grids.append(grid)
        return {
            "pixel_values_videos": (patches[0] if len(patches) == 1
                                    else np.concatenate(patches, axis=0)),
            "video_grid_thw": np.asarray(grids, dtype=np.int64),
        }

    __call__ = preprocess


def strip_template_spans(prompt: str) -> str:
    """Drop the vision markers the chat template writes around a single
    placeholder. The expansion adds exactly one pair."""
    for token in (IMAGE_TOKEN, VIDEO_TOKEN, AUDIO_TOKEN):
        prompt = prompt.replace(VISION_START + token + VISION_END, token)
    return prompt


def audio_expansion(tokens: int) -> str:
    """A clip's soft tokens between the vision markers, as a model with a
    vision tower wraps them."""
    return VISION_START + AUDIO_TOKEN * tokens + VISION_END


def image_expansion(soft_tokens: int) -> str:
    return VISION_START + IMAGE_TOKEN * soft_tokens + VISION_END


def soundtrack_bounds(tokens: int, indices: Sequence[int], fps: float,
                      temporal_patch_size: int, groups: int,
                      seconds_per_token: float) -> list:
    """Where a soundtrack of ``tokens`` encoder tokens is cut, one piece
    per frame group: each cut is at the time of a group's first frame.
    ``groups + 1`` rising values from 0 to ``tokens``."""
    starts = [int(i) / fps
              for i in list(indices)[::temporal_patch_size]][:groups]
    bounds = ([0]
              + [min(tokens, round(start / seconds_per_token))
                 for start in starts[1:]]
              + [tokens])
    return [max(bounds[:i + 1]) for i in range(len(bounds))]


def video_expansion(soft_tokens: int, timestamps: Sequence[float],
                    bounds: Optional[Sequence[int]] = None) -> str:
    """One group per temporal slice, inside one pair of markers. With
    ``bounds`` (:func:`soundtrack_bounds`), each group is followed by its
    piece of the soundtrack, as bare audio tokens."""
    groups = ""
    for i, t in enumerate(timestamps):
        groups += (f"<{t:.1f} seconds>" + VISION_START
                   + VIDEO_TOKEN * soft_tokens + VISION_END)
        if bounds is not None:
            groups += AUDIO_TOKEN * (bounds[i + 1] - bounds[i])
    return VISION_START + groups + VISION_END


def build_processor(tokenizer, image_processor: Lfm3ImageProcessor,
                    video_processor: Lfm3VideoProcessor,
                    feature_extractor=None):
    """``feature_extractor`` is the speech front end of a model with an
    audio encoder (gmlx.models.lfm3.audio), or None."""
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from gmlx._vendor.mlx_vlm.models.base import to_mlx
    from gmlx.upstream.single_bos_fix import KWARGS_REACH_TOKENIZER

    class Lfm3Processor(ProcessorMixin):
        attributes = ["image_processor", "tokenizer"]
        image_processor_class = "AutoImageProcessor"
        tokenizer_class = "AutoTokenizer"

        image_token = IMAGE_TOKEN
        video_token = VIDEO_TOKEN
        audio_token = AUDIO_TOKEN
        supports_multiple_audio = True

        def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
                self, images=None, text=None, videos=None, audio=None,
                fps=None, video_metadata=None, soundtracks=None, **kwargs):
            """``soundtracks`` holds one entry per video: its sound as
            mono samples at the encoder's rate, or None for a video
            without sound."""
            # The input pipeline hands over only the arguments this
            # signature names, so the video ones are named. The frame
            # times come from the metadata, and fps is not read.
            del fps
            if audio is not None and not isinstance(audio, (list, tuple)):
                audio = [audio]
            if soundtracks is not None and all(
                    track is None or len(track) == 0
                    for track in soundtracks):
                soundtracks = None
            if (audio or soundtracks) and self.feature_extractor is None:
                raise ValueError("this LFM3 model has no audio encoder")
            metadata = video_metadata
            if text is None:
                raise ValueError("LFM3 needs a text prompt")
            if isinstance(text, str):
                text = [text]
            text = [strip_template_spans(t) for t in text]
            if images is not None and not isinstance(images, (list, tuple)):
                images = [images]
            if videos is not None and not isinstance(videos, (list, tuple)):
                videos = [videos]
            for name, token, media in (("images", IMAGE_TOKEN, images),
                                       ("videos", VIDEO_TOKEN, videos),
                                       ("audio clips", AUDIO_TOKEN, audio)):
                found = sum(t.count(token) for t in text)
                given = len(media) if media else 0
                if found != given:
                    raise ValueError(
                        f"Found {found} {token} tokens in the text but "
                        f"received {given} {name}")

            data: dict = {}
            image_text: list = []
            if images:
                pre = self.image_processor.preprocess(images)
                data.update(pre)
                image_text = [
                    image_expansion(self.image_processor.soft_tokens(grid))
                    for grid in pre["image_grid_thw"]]
            if soundtracks is not None and len(soundtracks) != (
                    len(videos) if videos else 0):
                raise ValueError(
                    f"received {len(soundtracks)} soundtracks for "
                    f"{len(videos) if videos else 0} videos")
            tracks = [None if track is None or len(track) == 0
                      else self._clip(track)
                      for track in soundtracks or ()]
            clips = [self._clip(clip) for clip in audio or ()]
            fe = self.feature_extractor
            if clips or tracks:
                # The encoder's clips in prompt order: a video's
                # soundtrack sits at its video, between the clips.
                data.update(fe(self._in_prompt_order(text, clips, tracks)))
            video_text: list = []
            if videos:
                if metadata is None or len(metadata) != len(videos):
                    raise ValueError(
                        "LFM3 prompts carry frame timestamps: pass one "
                        "video_metadata entry per video, with the sampled "
                        "frame indices and the clip's fps")
                pre = self.video_processor.preprocess(videos)
                data.update(pre)
                for i, (grid, meta) in enumerate(
                        zip(pre["video_grid_thw"], metadata)):
                    track = tracks[i] if tracks else None
                    video_text.append(self._video_text(
                        grid, meta,
                        None if track is None else fe.tokens(track.size)))
            audio_text = [audio_expansion(fe.tokens(clip.size))
                          for clip in clips]
            text = self._expand(text, image_text, video_text, audio_text)

            kwargs.pop("return_tensors", None)
            return BatchFeature(
                data=to_mlx({**self.tokenizer(text, **kwargs), **data}))

        @staticmethod
        def _clip(samples) -> np.ndarray:
            clip = np.asarray(samples, dtype=np.float32)
            if clip.ndim != 1 or clip.size == 0:
                raise ValueError(
                    "LFM3 audio input takes mono clips with at least one "
                    f"sample, got shape {clip.shape}")
            return clip

        @staticmethod
        def _in_prompt_order(texts, clips, tracks) -> list:
            import re

            ordered = []
            clip, video = iter(clips), 0
            for t in texts:
                for m in re.finditer(
                        re.escape(VIDEO_TOKEN) + "|" + re.escape(AUDIO_TOKEN),
                        t):
                    if m.group(0) == AUDIO_TOKEN:
                        ordered.append(next(clip))
                        continue
                    if video < len(tracks) and tracks[video] is not None:
                        ordered.append(tracks[video])
                    video += 1
            return ordered

        def _video_text(self, grid, meta, soundtrack_tokens=None) -> str:
            fps = getattr(meta, "fps", None)
            indices = getattr(meta, "frames_indices", None)
            if indices is None or fps is None or not (
                    math.isfinite(fps) and fps > 0):
                raise ValueError(
                    "LFM3 prompts carry frame timestamps: video_metadata "
                    "must hold the sampled frame indices and a positive fps")
            fps = round(float(fps), 6)
            temporal = self.video_processor.temporal_patch_size
            stamps = frame_timestamps(indices, fps, temporal)[:int(grid[0])]
            bounds = None
            if soundtrack_tokens is not None:
                fe = self.feature_extractor
                bounds = soundtrack_bounds(
                    soundtrack_tokens, indices, fps, temporal, int(grid[0]),
                    fe.seconds_per_token)
            return video_expansion(
                self.video_processor.soft_tokens(grid), stamps, bounds)

        @staticmethod
        def _expand(texts, image_text, video_text, audio_text=()):
            """Replace each placeholder with its expansion, in order
            across the batch."""
            import re

            pattern = re.compile("|".join(map(
                re.escape, (IMAGE_TOKEN, VIDEO_TOKEN, AUDIO_TOKEN))))
            queues = {IMAGE_TOKEN: iter(image_text),
                      VIDEO_TOKEN: iter(video_text),
                      AUDIO_TOKEN: iter(audio_text)}
            return [pattern.sub(lambda m: next(queues[m.group(0)]), t)
                    for t in texts]

        def batch_decode(self, *args, **kwargs):
            return self.tokenizer.batch_decode(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.tokenizer.decode(*args, **kwargs)

        @property
        def model_input_names(self):
            return list(dict.fromkeys(
                list(self.tokenizer.model_input_names)
                + list(self.image_processor.model_input_names)
                + list(self.video_processor.model_input_names)))

    # __call__ hands **kwargs to self.tokenizer (single_bos_fix).
    setattr(Lfm3Processor, KWARGS_REACH_TOKENIZER, True)
    processor = Lfm3Processor(
        image_processor=image_processor, tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None))
    processor.video_processor = video_processor
    processor.feature_extractor = feature_extractor
    return processor
