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
# The extensions and headers that mlx-audio decodes with ffmpeg.
_FFMPEG_EXTS = ("m4a", "m4b", "mp4", "aac", "ogg", "opus", "webm", "caf")
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


def _miniaudio_decode(source, field: str, rate: int | None = None,
                      channels: int | None = None,
                      max_seconds: float | None = None) -> tuple[bytearray, int, int]:
    """16-bit samples, the sample rate and the channel count of ``source``,
    bytes or a file path, decoded by miniaudio at ``rate`` with ``channels``
    channels, each the file's own when not given, as mlx-audio decodes it.
    With ``max_seconds``, the decode stops after that many seconds."""
    import miniaudio

    if isinstance(source, str):
        info = miniaudio.get_file_info(source)
    else:
        info = _miniaudio_info(source)
    native = int(info.sample_rate)
    rate, channels = rate or native, channels or int(info.nchannels)
    for r in (native, rate):
        if not 0 < r <= mg.AUDIO_MAX_RATE:
            raise mg.audio_rate_refusal(field, r)
    stop = None
    if max_seconds is not None and math.ceil(max_seconds * rate) * channels \
            <= mg.AUDIO_MAX_SAMPLES:
        stop = 2 * math.ceil(max_seconds * rate) * channels
    # The length in the header refuses a long clip before the decode. A
    # header can give no length or a wrong one, so the decode stops at the
    # limit too.
    if stop is None and (math.ceil(int(info.num_frames) * rate / native) * channels
                         > mg.AUDIO_MAX_SAMPLES):
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
            if stop is not None and len(pcm) >= stop:
                del pcm[stop:]
                break
            if len(pcm) > limit:
                raise mg.audio_too_long(field, rate, channels)
    finally:
        stream.close()
    return pcm, rate, channels


def _downsample(samples, rate: int, target: int, channels: int):
    """``samples`` (16-bit, interleaved) at ``target`` Hz as normalized
    float32, through mlx-audio's chunked FIR resampler, as its reader
    downsamples."""
    import numpy as np
    from mlx_audio.audio_io import _RESAMPLE_CHUNK_DURATION_SECONDS
    from mlx_audio.resample import resample_audio_chunks

    frames = samples.reshape(-1, channels) if channels > 1 else samples
    step = max(1, int(_RESAMPLE_CHUNK_DURATION_SECONDS * rate))

    def chunks():
        for start in range(0, len(frames), step):
            chunk = frames[start:start + step].astype(np.float32)
            chunk /= 32768.0
            yield chunk

    return resample_audio_chunks(chunks(), rate, target, len(frames),
                                 chunk_duration_seconds=_RESAMPLE_CHUNK_DURATION_SECONDS)


def read_audio(file, always_2d: bool = False, dtype: str = "float64",
               sample_rate: int | None = None, nchannels: int | None = None,
               field: str = "audio", max_seconds: float | None = None):
    """Read audio as mlx-audio's ``audio_io.read`` does, from a file path or
    a BytesIO, at ``sample_rate`` with ``nchannels`` channels when given,
    and give the samples and the sample rate. The decode stops at
    :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples, or with
    ``max_seconds`` after that many seconds, and then gives the start of a
    longer clip."""
    import numpy as np

    if sample_rate is not None and sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if nchannels is not None and nchannels <= 0:
        raise ValueError(f"nchannels must be positive, got {nchannels}")
    if isinstance(file, (str, Path)):
        source = str(file)
        use_ffmpeg = Path(source).suffix.lstrip(".").lower() in _FFMPEG_EXTS
    elif isinstance(file, io.BytesIO):
        source = file.getvalue()
        use_ffmpeg = (source[4:8] == b"ftyp" or source[:4] == b"OggS"
                      or source[:4] == b"\x1a\x45\xdf\xa3" or source[:4] == b"caff")
    else:
        raise TypeError(f"Unsupported file type: {type(file)}")
    normalized = False
    if use_ffmpeg:
        samples, rate, channels = media_programs.decode(
            source, field, sample_rate=sample_rate, nchannels=nchannels,
            max_seconds=max_seconds)
    else:
        if isinstance(source, str):
            native = int(_file_info(source).sample_rate)
        else:
            native = int(_miniaudio_info(source).sample_rate)
        if sample_rate is not None and sample_rate < native:
            # mlx-audio decodes at the file's rate and downsamples through
            # its FIR resampler.
            pcm, rate, channels = _miniaudio_decode(source, field, None, nchannels,
                                                    max_seconds)
            if math.ceil(len(pcm) // 2 // channels * sample_rate / rate) * channels \
                    > mg.AUDIO_MAX_SAMPLES:
                raise mg.audio_too_long(field, sample_rate, channels)
            samples = _downsample(np.frombuffer(pcm, dtype=np.int16), rate,
                                  sample_rate, channels)
            rate, normalized = sample_rate, True
        else:
            pcm, rate, channels = _miniaudio_decode(source, field, sample_rate,
                                                    nchannels, max_seconds)
            samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1 and samples.ndim == 1:
        samples = samples.reshape(-1, channels)
    if dtype in ("float32", "float64"):
        samples = samples.astype(dtype, copy=normalized)
        if not normalized:
            samples /= 32768.0
    elif normalized:
        samples = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16)
        if dtype != "int16":
            samples = samples.astype(dtype)
    elif dtype != "int16":
        samples = samples.astype(dtype)
    if always_2d and samples.ndim == 1:
        samples = samples[:, np.newaxis]
    return samples, rate


def _file_info(path: str):
    import miniaudio

    return miniaudio.get_file_info(path)


def ffmpeg_resample(audio, rate: int, target: int):
    """``audio`` resampled from ``rate`` to ``target`` as FFmpeg's resampler
    does with its defaults (libswresample: a Kaiser-windowed sinc, beta 9,
    cutoff 0.97, 32 taps), for a pair of rates it resamples exactly. Its
    phase count caps that at 1024 output phases; past it, mlx-audio's
    resampler runs."""
    import numpy as np
    from scipy.signal import resample_poly

    g = math.gcd(rate, target)
    up, down = target // g, rate // g
    if up > 1024:
        from mlx_audio.utils import resample_audio

        return resample_audio(audio, rate, target)
    # swr's cutoff 0.97 of the lower Nyquist, capped at the input's own
    # Nyquist when upsampling; 32 taps per input sample at that band.
    factor = min(up / down * 0.97, 1.0)
    half = math.ceil(16 / factor) * up
    band = factor / up
    taps = band * np.sinc(band * np.arange(-half, half + 1)) * np.kaiser(2 * half + 1, 9.0)
    taps /= taps.sum()
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < 2:
        return resample_poly(x, up, down, window=taps)
    # swr mirrors the clip for the filter to read past its ends: the start
    # without its first sample, the end with its last.
    pad = math.ceil(math.ceil(16 / factor) / down) * down
    x = np.concatenate([np.pad(x, (pad, 0), mode="reflect"),
                        np.pad(x, (0, pad), mode="symmetric")[len(x):]])
    y = resample_poly(x, up, down, window=taps)
    start = pad * up // down
    return y[start:start + math.ceil((len(x) - 2 * pad) * up / down)]


def load_audio(file, sr: int, field: str = "audio",
               max_seconds: float | None = None, ffmpeg: bool = False):
    """mlx-vlm's ``load_audio`` for a BytesIO or an array: mono float32
    samples at ``sr``. Audio that resampling to ``sr`` would take past
    :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples is refused before
    the resample. With ``max_seconds``, only the start of a longer clip is
    decoded and resampled. With ``ffmpeg``, a stereo clip mixes down and the
    clip resamples as FFmpeg does by default, as torchcodec decodes for
    transformers' ``load_audio``: the channels summed at 0.5 ** 0.5 each,
    then :func:`ffmpeg_resample`. Other channel counts take the mean.

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
    audio, rate = read_audio(file, dtype="float32", field=field,
                             max_seconds=max_seconds)
    if ffmpeg and audio.ndim > 1 and audio.shape[1] == 2:
        audio = (audio[:, 0] + audio[:, 1]) * np.float32(0.5 ** 0.5)
    elif audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sr:
        if math.ceil(len(audio) * sr / rate) > mg.AUDIO_MAX_SAMPLES:
            raise mg.audio_too_long(field, sr, 1)
        audio = ffmpeg_resample(audio, rate, sr) if ffmpeg else resample_audio(audio, rate, sr)
    return np.asarray(audio, dtype=np.float32)


# Video

# The 400 for a video when the server has no decoder.
NO_VIDEO_DECODER = ("video input needs ffmpeg or OpenCV: install one with "
                    "`brew install ffmpeg` or `{hint}`")


def read_video(video_path: str, sampling=None, frame_sampler=None,
               **sampling_kwargs):
    """Read a video file as mlx-vlm's ``load_video`` does: a (T, C, H, W)
    array of the sampled frames and their ``VideoMetadata``. The frames
    count toward the request's decoded pixels before the first one is
    read.

    With ffmpeg and ffprobe on PATH, the frames and the metadata are the
    ones torchcodec gives transformers' video processors. Otherwise OpenCV
    reads the video, and raises :class:`~.patches.media_gate.MediaRefused`
    when it is not installed either."""
    from dataclasses import fields

    from gmlx._vendor.mlx_vlm.utils import DEFAULT_VIDEO_SAMPLING, VideoSampling

    if sampling_kwargs:
        unknown = set(sampling_kwargs) - {f.name for f in fields(VideoSampling)}
        if unknown:
            raise TypeError("load_video() got unexpected keyword arguments: "
                            f"{sorted(unknown)}")
        sampling = (sampling or VideoSampling()).merge(VideoSampling(**sampling_kwargs))
    resolved = (sampling or VideoSampling()).merge(DEFAULT_VIDEO_SAMPLING)
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if media_programs.find("ffmpeg") and media_programs.find("ffprobe"):
        return _read_video_ffmpeg(video_path, resolved, frame_sampler)
    from gmlx.commands.extras import install_hint, opencv

    try:
        cv2 = opencv("video input")
    except ImportError:
        raise mg.MediaRefused(NO_VIDEO_DECODER.format(hint=install_hint("video"))) from None
    return _read_video_opencv(cv2, video_path, resolved, frame_sampler)


def _pick_frames(resolved, total_frames: int, video_fps: float, width: int,
                 height: int, duration: float, frame_sampler):
    """The indices of the frames to read, as mlx-vlm's ``load_video`` picks
    them, after the pixel checks and the charge for them."""
    import numpy as np
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    fps, nframes = resolved.fps, resolved.nframes
    min_frames, max_frames = resolved.min_frames, resolved.max_frames
    frame_factor = resolved.frame_factor

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
        if fps is None or max_frames is None:
            # The defaults set both; a sampling that clears one has no
            # frame count to give.
            raise ValueError("video sampling needs fps and max_frames")
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
    return indices


def _read_video_opencv(cv2, video_path: str, resolved, frame_sampler):
    import numpy as np
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {os.path.basename(video_path)}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps
        indices = _pick_frames(resolved, total_frames, video_fps, width, height,
                               duration, frame_sampler)
        n = len(indices)
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
    if not count or frames is None:
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


# The most text an ffprobe packet list may take: about 15 bytes a packet.
_PROBE_LIMIT = 64 << 20


def _probe_video(ffprobe: str, video_path: str):
    """``(frames, fps, duration, width, height, turns)`` of the first video
    stream, as torchcodec's exact mode gives them: the frame count from the
    packets, the duration from the first packet's time to the latest end
    of a packet, and the rate from the two. The size is the stored one, and
    ``turns`` the quarter turns counterclockwise that show the frame."""
    import json

    name = os.path.basename(video_path)
    stream = media_programs._run(
        [ffprobe, "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,time_base:stream_side_data=rotation", "-of", "json",
         "file:" + video_path], None)
    try:
        streams = json.loads(stream.stdout or b"{}").get("streams") or []
    except ValueError:
        streams = []
    if stream.returncode or not streams:
        raise ValueError(f"Cannot open video: {name}")
    info = streams[0]
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    rotation = next((float(d["rotation"]) for d in info.get("side_data_list") or []
                     if "rotation" in d), 0.0)
    turns = round(rotation / 90) % 4
    num, _, den = str(info.get("time_base") or "1/1").partition("/")
    time_base = int(num) / int(den or 1)

    run = media_programs._run_bounded(
        [ffprobe, "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
         "-show_entries",
         "packet=pts,dts,duration", "-of", "csv=p=0", "file:" + video_path], None, _PROBE_LIMIT)
    if run is None:
        raise ValueError(f"Too many frames in video: {name}")
    code, out, _ = run
    count, begin, end = 0, math.inf, -math.inf
    for line in out.decode("ascii", "replace").splitlines():
        pts, dts, length = (line.split(",") + ["", "", ""])[:3]
        start = pts if pts.lstrip("-").isdigit() else dts
        if not start.lstrip("-").isdigit():
            continue
        count += 1
        t = int(start)
        begin = min(begin, t)
        end = max(end, t + (int(length) if length.isdigit() else 0))
    if code or not count or not (width and height):
        raise ValueError(f"Cannot read video: {name}")
    duration = (end - begin) * time_base
    fps = count / duration if duration > 0 else 1.0
    return count, fps, duration, width, height, turns


def _select(frames: list[int]) -> str:
    """ffmpeg's select expression for ``frames``. Its parser refuses a sum
    of more than 100 terms, so the terms nest in groups of 16."""
    terms = [f"eq(n\\,{i})" for i in frames]

    def tree(ts: list[str]) -> str:
        if len(ts) <= 16:
            return "+".join(ts)
        size = -(-len(ts) // 16)
        return "+".join(f"({tree(ts[i:i + size])})" for i in range(0, len(ts), size))
    return tree(terms)


def _read_video_ffmpeg(video_path: str, resolved, frame_sampler):
    import numpy as np
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    ffprobe = media_programs.program("ffprobe")
    ffmpeg = media_programs.program("ffmpeg")
    total_frames, video_fps, duration, width, height, turns = _probe_video(ffprobe, video_path)
    shown = (height, width) if turns % 2 else (width, height)
    indices = _pick_frames(resolved, total_frames, video_fps, *shown, duration,
                           frame_sampler)
    n = len(indices)

    # ffmpeg decodes the stream once and keeps the picked frames, each once;
    # a sampler may pick a frame twice. The decoder opens local files only.
    # torchcodec turns a frame after its conversion to RGB, and the ffmpeg
    # CLI before, so the frames come unturned and turn here.
    wanted = sorted(set(indices.tolist()))
    pick = _select(wanted)
    frame_bytes = width * height * 3
    run = media_programs._run_bounded(
        [ffmpeg, "-v", "error", "-nostdin", "-protocol_whitelist", "file",
         "-noautorotate", "-i", "file:" + video_path, "-map", "0:v:0",
         "-vf", f"select='{pick}'", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        None, len(wanted) * frame_bytes)
    if run is None:
        raise ValueError("The frames of the video change size.")
    code, out, errors = run
    got = len(out) // frame_bytes
    if code or not got:
        detail = errors.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(f"Cannot read video: {detail[-1] if detail else 'no frames'}")
    decoded = np.frombuffer(bytes(out[:got * frame_bytes]), dtype=np.uint8).reshape(
        got, height, width, 3)
    # A file with fewer frames than its packets ends part-way.
    position = {f: k for k, f in enumerate(wanted[:got])}
    picked = indices.tolist()
    count = next((k for k, i in enumerate(picked) if i not in position), n)
    kept = picked[:count]
    frames = np.rot90(decoded[[position[i] for i in kept]], turns, axes=(1, 2))
    video_np = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    metadata = VideoMetadata(total_num_frames=total_frames, fps=video_fps,
                             frames_indices=kept, height=shown[1], width=shown[0],
                             duration=duration)
    return video_np, metadata
