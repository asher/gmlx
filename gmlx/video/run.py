"""The parts of a clip's run that the video command and the server's
video jobs share: the anchors read and placed before any model loads,
and the decode of the latents into an MP4.

The command loads each model of a run itself, and a job runs on a
served entry's pipeline, so the encode and the denoise stay with each.
Both decode the same way: the audio first, with the audio VAE alone,
then the video a chunk of frames at a time into ffmpeg, with the video
VAE alone, so a clip's latents give the same file from either.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections.abc import Callable

import numpy as np

# A guide's sound is read to a second past the clip's audio, more than the
# audio encode's halo (12 latents) past the latents an anchor keeps.
GUIDE_AUDIO_PAST_S = 1


@dataclasses.dataclass(frozen=True)
class Anchor:
    """An anchor a run asks for, read and placed before any model loads:
    its flag (or field) and path, its placement and planned rows, and
    what it encodes from: a still at the canvas size, a video file whose
    frames are read when it encodes, its sound, or both."""
    flag: str
    path: str
    placed: object                  # anchors.Placement
    rows: object                    # layout.AnchorRows
    still: object = None            # (H, W, 3) uint8
    video: str | None = None
    wave: object = None             # (2, samples) float32
    ffmpeg: str | None = None
    label: str = ""                 # how its refusals name it


def is_still(path: str) -> bool:
    """Whether Pillow opens ``path`` as an image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path):
            return True
    except UnidentifiedImageError:
        return False


def read_anchors(fam, s: dict, *, first: str | None = None,
                 last: str | None = None, guides=(),
                 guide_no_audio: bool = False, ffmpeg: str | None = None,
                 names: tuple = ("--first-frame", "--last-frame"),
                 paths: bool = True) -> tuple[list, list]:
    """The keyframes, (H, W, 3) uint8 at the canvas in prompt order, and
    each anchor as :class:`Anchor`, before any model loads: the keyframe
    images ``first`` and ``last``, and ``guides``, (flag, path, frame)
    of ``--guide`` or ``--guide-audio``, of the clip of settings ``s``.
    ``names`` are the keyframes' flags as the refusals name them, each
    with its path unless ``paths`` is False; a guide may carry its own
    name as a fourth item. A guide video's frames are counted here and
    read when it encodes."""
    from gmlx.video import anchors, images, io
    from gmlx.video.layout import AnchorRows
    from gmlx.video.vae.h3_audio import HOP, SAMPLE_RATE

    frames, w, h = s["frames"], s["width"], s["height"]
    audio_t = fam.latent_shapes(frames, w, h)[1][3]
    samples = audio_t * HOP + GUIDE_AUDIO_PAST_S * SAMPLE_RATE
    exe = ffmpeg
    keyframes: list = []
    out: list = []

    def add(flag, path, frame, *, n=0, wave=None, most=None, label, **kw):
        try:
            if n == most:
                # Read to its bound, the video holds more than fits.
                raise ValueError(
                    f"a guide clip of {anchors.guide_frames(n)} or more "
                    f"frames at frame {frame} does not fit in the clip's "
                    f"{frames} frames")
            p = anchors.place(frame, frames, audio_t,
                              fam.timeline.frame_rescale, video_frames=n,
                              audio=wave is not None)
        except ValueError as e:
            raise ValueError(f"{label}: {e}") from None
        video_t = (0 if not p.video_frames else 1 if p.video_frames == 1
                   else fam.latent_shapes(p.video_frames, w, h)[0][2])
        sound = (0 if p.audio_t is None
                 else min(p.audio_t, wave.shape[1] // HOP))
        out.append(Anchor(flag, path, p, AnchorRows(p.frame, video_t, sound),
                          wave=wave, label=label, **kw))

    for flag, path, frame, resize in (
            (names[0], first, 0, images.stretch),
            (names[1], last, frames - 1, images.cover)):
        if path:
            im = resize(images.read(os.path.expanduser(path), exe), w, h)
            keyframes.append(im)
            add(flag, path, frame, n=1, still=im,
                label=f"{flag} {path}" if paths else flag)
    for g in guides or ():
        flag, path, frame = g[:3]
        label = g[3] if len(g) > 3 else f"{flag} {path}"
        file = os.path.expanduser(path)
        if flag == "--guide" and is_still(file):
            add(flag, path, frame, n=1, label=label,
                still=images.cover(images.read(file, exe), w, h))
            continue
        if exe is None:
            raise io.FfmpegError(
                f"{label}: gmlx reads video and audio files through "
                "ffmpeg, which is not on PATH: install ffmpeg (brew install "
                "ffmpeg)")
        at = frame + frames if frame < 0 else frame
        if not 0 <= at < frames:
            add(flag, path, frame, n=1, label=label)  # refuses a frame off the clip
        wave = None
        if flag == "--guide-audio" or not guide_no_audio:
            wave = io.guide_audio(file, rate=SAMPLE_RATE, samples=samples,
                                  ffmpeg=exe, hop=HOP,
                                  optional=flag == "--guide")
        if flag == "--guide-audio":
            add(flag, path, frame, wave=wave, label=label)
            continue
        # 17 frames past those left from the frame tell a guide that does
        # not fit from one that does.
        most = frames - at + 17
        add(flag, path, frame, n=io.guide_length(file, most=most, ffmpeg=exe),
            most=most, wave=wave, video=file, ffmpeg=exe, label=label)
    return keyframes, out


def upsample_system(fam, path: str | None, *, frames: int | None,
                    first: bool = False, last: bool = False) -> str:
    """The upsampler's system prompt for a clip of ``frames``, None when
    the length is not known, with these keyframes: the family's lines
    around the text of the file at ``path`` (the entry's
    ``upsample_prompt``), read on each call. Raises ``UpsampleError`` with
    the family's refusal when there is none."""
    from gmlx.image.upsample import system_prompt

    return fam.upsample_system(system_prompt(fam, path), frames, first, last)


def decode_and_mux(xv, xa, *, video_vae: Callable, audio_vae: Callable,
                   frames: int, width: int, height: int, fps: int,
                   rate: int, out: str | None, ffmpeg: str | None,
                   frames_dir: str | None = None,
                   between: Callable | None = None,
                   log: Callable | None = None) -> dict:
    """Decode the latents ``xv`` and ``xa`` of a clip: the audio through
    the model ``audio_vae()`` loads, which goes before the video VAE of
    ``video_vae()`` decodes the frames a chunk at a time into the MP4
    ``out`` (written as ``out.part``, then renamed) and the PNGs of
    ``frames_dir``. ``between(k)`` runs after each chunk, with ``k``
    frames decoded, and may raise to stop: the MP4 is then aborted and
    its part removed. Returns {"audio_s", "decode_s"}."""
    import time

    import mlx.core as mx

    from gmlx.video import io

    def release():
        import gc

        gc.collect()
        mx.clear_cache()

    times = {}
    t0 = time.perf_counter()
    avae = audio_vae()
    try:
        # One length for the MP4 and the frames' audio.wav.
        wave = io.fit_audio(np.array(avae.decode(xa)[0]), frames=frames,
                            fps=fps, rate=rate)
    finally:
        del avae
        release()
    times["audio_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    vae = video_vae()
    try:
        if frames_dir is not None:
            os.makedirs(os.path.expanduser(frames_dir), exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gmlx-video-") as tmp:
            wav = io.write_wav(os.path.join(tmp, "audio.wav"), wave, rate)
            mux = None
            if out is not None:
                os.makedirs(os.path.dirname(os.path.abspath(out)),
                            exist_ok=True)
                mux = io.Mux(out, width=width, height=height, fps=fps,
                             wav=wav, ffmpeg=ffmpeg or "")
            try:
                n = 0
                for chunk in vae.decode_frames(xv):
                    got = np.array(chunk[0])
                    if mux is not None:
                        mux.write(got)
                    if frames_dir is not None:
                        n = io.write_frames(os.path.expanduser(frames_dir),
                                            got, n)
                    done = mux.frames if mux else n
                    if log is not None:
                        log(f"decoded {done} of {frames} frames")
                    if between is not None:
                        between(done)
                if mux is not None:
                    mux.close()
            except BaseException:
                if mux is not None:
                    mux.abort()
                raise
            if frames_dir is not None:
                io.write_wav(os.path.join(os.path.expanduser(frames_dir),
                                          "audio.wav"), wave, rate)
    finally:
        del vae
        release()
    times["decode_s"] = round(time.perf_counter() - t0, 2)
    return times
