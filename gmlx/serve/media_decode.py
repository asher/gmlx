"""Decoders for request media that stop before a large allocation.

A request holds at most :data:`~.patches.media_gate.MEDIA_MAX_BYTES` of
each image, audio clip or video, but a small file can decode to much more:
a PNG of one color to a 64-megapixel image, a FLAC file of silence to hours
of samples, a still video to hundreds of large frames. These decoders take
the place of the stock ones in mlx-audio and mlx-vlm and refuse such media
before they decode it, or stop the decode at the limit:

- :func:`read_audio` stops at
  :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples, whatever the
  file's header says, and refuses a sample rate over
  :data:`~.patches.media_gate.AUDIO_MAX_RATE`.
- :func:`load_audio` also refuses a clip that resampling to the model's
  rate would take past that count.
- :func:`read_video` and :func:`charge`, which the image loader calls with
  each image's size from its header, keep the images and video frames of
  one request under :data:`~.patches.media_gate.MEDIA_MAX_TOTAL_PIXELS`
  pixels, and its images, audio clips and videos under
  :data:`~.patches.media_gate.MEDIA_MAX_ITEMS`.

:func:`decode_pass` holds the count for one request. The server decodes the
media of one request at a time, so the decoded media of all requests
together stay within these limits. The media gate makes the same count from
the request body before anything decodes. These counts are the backstop for
media that the gate cannot see, such as a file in the media folder.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import threading
from pathlib import Path

from gmlx.serve import media_programs
from gmlx.serve.patches import media_gate as mg

# The formats that mlx-audio decodes with ffmpeg, by file name extension.
_FFMPEG_EXTS = ("m4a", "aac", "ogg", "opus", "webm")
# Frames read from miniaudio at a time.
_CHUNK_FRAMES = 1 << 16


# The decode pass of one request

class _Pass:
    """The media of one request that decoded so far: the items, and the
    pixels of the images and video frames."""

    __slots__ = ("items", "pixels")

    def __init__(self) -> None:
        self.items = 0
        self.pixels = 0


_local = threading.local()
# Held while a request's media decode, so that one request at a time holds
# decoded media.
_DECODE_LOCK = threading.Lock()


@contextlib.contextmanager
def decode_pass():
    """Count the decoded pixels of one request's media, and decode the media
    of one request at a time. A pass inside a pass on the same thread is
    the outer pass."""
    if getattr(_local, "current", None) is not None:
        yield _local.current
        return
    with _DECODE_LOCK:
        _local.current = _Pass()
        try:
            yield _local.current
        finally:
            _local.current = None


def charge(what: str, pixels: int = 0, item: bool = True) -> None:
    """Count ``what``, an item that decodes to ``pixels`` pixels, toward
    the request's media, before it decodes. Raises
    :class:`~.patches.media_gate.MediaRefused` past a limit. Outside a pass,
    ``what`` is counted alone."""
    current = getattr(_local, "current", None) or _Pass()
    items = current.items + (1 if item else 0)
    if items > mg.MEDIA_MAX_ITEMS:
        raise mg.too_many_items(items)
    total = current.pixels + pixels
    if total > mg.MEDIA_MAX_TOTAL_PIXELS:
        raise mg.too_many_request_pixels(what, total)
    current.items, current.pixels = items, total


# Audio

def _miniaudio_info(data: bytes):
    import miniaudio
    from mlx_audio.audio_io import _detect_format_from_bytes

    fmt = _detect_format_from_bytes(data)
    readers = {"wav": miniaudio.wav_get_info, "mp3": miniaudio.mp3_get_info,
               "flac": miniaudio.flac_get_info, "vorbis": miniaudio.vorbis_get_info}
    if fmt not in readers:
        raise ValueError(f"Unsupported format: {fmt}")
    return readers[fmt](data)


def _miniaudio_decode(source, field: str) -> tuple[bytearray, int, int]:
    """16-bit samples, the sample rate and the channel count of ``source``,
    bytes or a file path, decoded by miniaudio at the file's own rate and
    channel count, as mlx-audio decodes it."""
    import miniaudio

    if isinstance(source, str):
        info = miniaudio.get_file_info(source)
    else:
        info = _miniaudio_info(source)
    rate, channels = int(info.sample_rate), int(info.nchannels)
    if not 0 < rate <= mg.AUDIO_MAX_RATE:
        raise mg.audio_rate_refusal(field, rate)
    # The length in the header refuses a long clip before the decode. A
    # header can give no length or a wrong one, so the decode stops at the
    # limit too.
    if int(info.num_frames) * channels > mg.AUDIO_MAX_SAMPLES:
        raise mg.audio_too_long(field, rate, channels)
    if isinstance(source, str):
        stream = miniaudio.stream_file(source, miniaudio.SampleFormat.SIGNED16, channels,
                                       rate, frames_to_read=_CHUNK_FRAMES)
    else:
        stream = miniaudio.stream_memory(source, miniaudio.SampleFormat.SIGNED16,
                                         channels, rate, frames_to_read=_CHUNK_FRAMES)
    limit = 2 * mg.AUDIO_MAX_SAMPLES
    pcm = bytearray()
    try:
        for chunk in stream:
            pcm += chunk
            if len(pcm) > limit:
                raise mg.audio_too_long(field, rate, channels)
    finally:
        stream.close()
    return pcm, rate, channels


def read_audio(file, always_2d: bool = False, dtype: str = "float64",
               field: str = "audio"):
    """Read audio as mlx-audio's ``audio_io.read`` does, from a file path or
    a BytesIO, and give the samples and the sample rate. The decode stops at
    :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples."""
    import numpy as np

    if isinstance(file, (str, Path)):
        source = str(file)
        use_ffmpeg = Path(source).suffix.lstrip(".").lower() in _FFMPEG_EXTS
    elif isinstance(file, io.BytesIO):
        source = file.getvalue()
        use_ffmpeg = (source[4:8] == b"ftyp" or source[:4] == b"OggS"
                      or source[:4] == b"\x1a\x45\xdf\xa3")
    else:
        raise TypeError(f"Unsupported file type: {type(file)}")
    if use_ffmpeg:
        samples, rate, channels = media_programs.decode(source, field)
    else:
        pcm, rate, channels = _miniaudio_decode(source, field)
        samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    if dtype in ("float32", "float64"):
        samples = samples.astype(dtype)
        samples /= 32768.0
    elif dtype != "int16":
        samples = samples.astype(dtype)
    if always_2d and samples.ndim == 1:
        samples = samples[:, np.newaxis]
    return samples, rate


def load_audio(file, sr: int, field: str = "audio"):
    """mlx-vlm's ``load_audio`` for a BytesIO or an array: mono float32
    samples at ``sr``. Audio that resampling to ``sr`` would take past
    :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples is refused before
    the resample.

    mlx-vlm resamples a clip of several channels along the channel axis and
    mixes it down after, which leaves it at its own rate and takes minutes
    for a long clip. This mixes the channels down first, then resamples the
    one channel. A mono clip gives the same samples as in mlx-vlm."""
    import numpy as np

    if isinstance(file, np.ndarray):
        audio = file.astype(np.float32, copy=False)
        return audio.mean(axis=1) if audio.ndim > 1 else audio
    from mlx_audio.utils import resample_audio

    charge(field)
    audio, rate = read_audio(file, dtype="float32", field=field)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sr:
        if math.ceil(len(audio) * sr / rate) > mg.AUDIO_MAX_SAMPLES:
            raise mg.audio_too_long(field, sr, 1)
        audio = resample_audio(audio, rate, sr)
    return np.asarray(audio, dtype=np.float32)


# Video

def read_video(video_path: str, sampling=None, frame_sampler=None,
               **sampling_kwargs):
    """Read a video file as mlx-vlm's ``load_video`` does: a (T, C, H, W)
    array of the sampled frames and their ``VideoMetadata``. The frames
    count toward the request's decoded pixels before the first one is
    read."""
    from dataclasses import fields

    import cv2
    import numpy as np
    from mlx_vlm.utils import DEFAULT_VIDEO_SAMPLING, VideoMetadata, VideoSampling

    if sampling_kwargs:
        unknown = set(sampling_kwargs) - {f.name for f in fields(VideoSampling)}
        if unknown:
            raise TypeError("load_video() got unexpected keyword arguments: "
                            f"{sorted(unknown)}")
        sampling = (sampling or VideoSampling()).merge(VideoSampling(**sampling_kwargs))
    resolved = (sampling or VideoSampling()).merge(DEFAULT_VIDEO_SAMPLING)
    fps, nframes = resolved.fps, resolved.nframes
    min_frames, max_frames = resolved.min_frames, resolved.max_frames
    frame_factor = resolved.frame_factor

    if video_path.startswith("file://"):
        video_path = video_path[7:]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {os.path.basename(video_path)}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps

        def _round(n):
            return round(n / frame_factor) * frame_factor

        def _floor(n):
            return math.floor(n / frame_factor) * frame_factor

        def _ceil(n):
            return math.ceil(n / frame_factor) * frame_factor

        sampled = False
        if nframes is not None:
            n = _round(nframes)
            indices = np.linspace(0, total_frames - 1, n).round().astype(int)
        elif frame_sampler is not None:
            sampled = True
            source = VideoMetadata(
                total_num_frames=total_frames, fps=video_fps,
                frames_indices=list(range(total_frames)), width=width,
                height=height, duration=duration)
            indices = np.asarray(
                frame_sampler(source, fps=fps, max_frames=max_frames),
                dtype=int).reshape(-1)
            n = len(indices)
        else:
            lo = _ceil(min_frames)
            hi = _floor(min(max_frames, total_frames))
            n = total_frames / video_fps * fps
            n = min(max(n, lo), hi, total_frames)
            n = _floor(n)
            indices = np.linspace(0, total_frames - 1, n).round().astype(int)
        if not sampled and not (frame_factor <= n <= total_frames):
            raise ValueError(f"nframes must be in [{frame_factor}, {total_frames}], "
                             f"got {n}.")
        if n == 0 or np.any(indices < 0) or np.any(indices >= total_frames):
            raise ValueError("Frame indices must be within a non-empty "
                             f"{total_frames}-frame video.")
        n = int(n)
        if width * height > mg.MEDIA_MAX_PIXELS:
            raise mg.too_many_pixels("video frame", width, height)
        charge(f"video of {n} frames of {width}x{height} pixels", n * width * height)
        frames = None
        count = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frames is None:
                h, w = rgb.shape[:2]
                if (h, w) != (height, width):
                    # The header gave no size, or another one.
                    if w * h > mg.MEDIA_MAX_PIXELS:
                        raise mg.too_many_pixels("video frame", w, h)
                    charge(f"video of {n} frames of {w}x{h} pixels",
                           n * (w * h - width * height), item=False)
                frames = np.empty((n, *rgb.shape), dtype=rgb.dtype)
            elif rgb.shape != frames.shape[1:]:
                raise ValueError("The frames of the video change size.")
            frames[count] = rgb
            count += 1
    finally:
        cap.release()
    if not count:
        raise ValueError("No frames read from the video.")
    video_np = np.transpose(frames[:count], (0, 3, 1, 2))
    metadata = VideoMetadata(
        total_num_frames=total_frames,
        fps=video_fps,
        # The frames returned, where a read failed part-way.
        frames_indices=indices[:count].tolist(),
        height=int(video_np.shape[2]),
        width=int(video_np.shape[3]),
        duration=duration,
    )
    return video_np, metadata
