"""Decode the media of an embeddings request, on the request thread.

Each image, audio clip and video decodes to the arrays that
:mod:`gmlx.models.embedders.gemma_embedding2_processor` takes, inside one
:func:`~.media_decode.decode_pass`, so the request's media count against the
server's media limits. The towers run later, on the service's worker.

- An image decodes as transformers' ``load_image`` does: EXIF orientation
  applied, then RGB.
- An audio clip decodes to mono 16 kHz samples. It is cut at 30 s, as the
  reference processor cuts it, and the decoder stops just past the cut.
  When a request lifts the cut, the decoder stops past the longest clip
  that fits one item, and a clip that reaches it is refused.
- A video's frames are sampled as the model's video processor samples them:
  1 frame per second, at most 32.
"""

from __future__ import annotations

import io

import numpy as np

from gmlx.models.embedders import gemma_embedding2_processor as proc

from .embeddings_errors import EmbeddingsRequestError
from .embeddings_request import MediaRef

# Decoded past the cut, so the resampler's edge leaves the kept samples as a
# decode of the whole clip gives them.
_MARGIN_S = 0.25

_WANT = {"image": "a base64 data:image/... URI",
         "audio": "base64 audio or a data:audio/... URI",
         "video": "a base64 data:video/... URI"}


def lifted_cut_samples(max_tokens: int) -> int:
    """The longest clip, in 16 kHz samples, whose soft tokens and markers fit
    an item of ``max_tokens`` tokens with BOS and EOS."""
    # 40 ms per soft token after the two stride-2 subsamplings.
    return (max_tokens - 4) * proc.AUDIO_RATE * 40 // 1000


def _bytes(ref: MediaRef) -> bytes:
    from . import media_sinks

    if ref.data is not None:
        if len(ref.data) > media_sinks.mg.MEDIA_MAX_BYTES:
            raise media_sinks.mg._too_large(ref.field)
        return ref.data
    assert ref.ref is not None
    if ref.path:
        return media_sinks.read_media_file(ref.ref, ref.field, _WANT[ref.kind])
    return media_sinks.reference_bytes(ref.ref, ref.kind, ref.field, _WANT[ref.kind])


def _image(ref: MediaRef) -> proc.Image:
    from PIL import ImageOps

    from . import media_sinks

    image = media_sinks.open_image(_bytes(ref), ref.field)
    try:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return proc.Image(np.asarray(rgb, dtype=np.uint8))
    finally:
        image.close()


def _audio(ref: MediaRef, options: proc.Options, max_tokens: int) -> proc.Audio:
    from . import media_decode

    if options.audio_cut:
        keep = proc.AUDIO_CUT
    else:
        keep = lifted_cut_samples(max_tokens)
    samples = media_decode.load_audio(
        io.BytesIO(_bytes(ref)), proc.AUDIO_RATE, ref.field,
        max_seconds=keep / proc.AUDIO_RATE + _MARGIN_S)
    if len(samples) > keep:
        if not options.audio_cut:
            raise EmbeddingsRequestError(
                400, f"{ref.field} is longer than {keep // proc.AUDIO_RATE} s, the "
                     "longest clip one input holds")
        samples = np.array(samples[:keep])
    return proc.Audio(np.ascontiguousarray(samples, dtype=np.float32))


def _video(ref: MediaRef) -> proc.Video:
    from gmlx._vendor.mlx_vlm.utils import VideoSampling

    from . import media_decode, media_sinks

    sampling = VideoSampling(fps=proc.VIDEO_FPS, max_frames=proc.VIDEO_MAX_FRAMES)

    def read(path: str):
        return media_decode.read_video(path, sampling=sampling,
                                       frame_sampler=proc.frame_sampler)

    if ref.data is None and ref.ref is not None and not ref.ref.startswith("data:") \
            and not media_sinks.mg._is_url(ref.ref):
        # A video in the media folder is read where it is.
        import os

        fd, shown = media_sinks._open_media_file(ref.ref, ref.field, _WANT["video"])
        try:
            head = os.pread(fd, 16, 0)
        finally:
            os.close(fd)
        media_sinks._check_container(head, ref.field)
        frames, _ = read(shown)
    else:
        data = _bytes(ref)
        media_sinks._check_container(data, ref.field)
        with media_sinks._private_file(data, ".video") as path:
            frames, _ = read(path)
    return proc.Video(np.ascontiguousarray(frames.transpose(0, 2, 3, 1)))


def decode(ref: MediaRef, options: proc.Options, max_tokens: int):
    """The decoded :class:`proc.Image`, :class:`proc.Audio` or
    :class:`proc.Video` of ``ref``. Call inside a
    :func:`~.media_decode.decode_pass`. Raises
    :class:`EmbeddingsRequestError` (400) for media gmlx cannot read."""
    from .patches.media_gate import MediaRefused

    try:
        if ref.kind == "image":
            return _image(ref)
        if ref.kind == "audio":
            return _audio(ref, options, max_tokens)
        return _video(ref)
    except EmbeddingsRequestError:
        raise
    except MediaRefused as exc:
        raise EmbeddingsRequestError(400, str(exc)) from None
    except (ValueError, RuntimeError, OSError) as exc:
        raise EmbeddingsRequestError(
            400, f"{ref.field} cannot be read as {ref.kind}: {exc}") from None


def media_tokens(parts: list, options: proc.Options) -> int:
    """The tokens an item's decoded media take, markers included."""
    total = 0
    for p in parts:
        if isinstance(p, str):
            continue
        m = proc.media_of(p, options)
        total += m.frames * (m.tokens + 2)
    return total
