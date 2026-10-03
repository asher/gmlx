"""gmlx/serve/media_decode.py and the media limits of one request: a small
file that decodes to a large image, hours of audio or many video frames is
refused before the large allocation, on every route that decodes media."""

from __future__ import annotations

import base64
import importlib
import io
import os
import struct
import threading
import tracemalloc
import wave
import zlib

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gmlx.container import settings
from gmlx.serve import media_decode as md
from gmlx.serve import media_programs, media_sinks, stt
from gmlx.serve.patches import media_gate as mg
from gmlx.serve.patches.media_gate import MediaRefused

_UTILS = importlib.import_module("mlx_vlm.utils")
_AUDIO_IO = importlib.import_module("mlx_audio.audio_io")


# Crafted media

def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _utf8_number(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    if n < 0x800:
        return bytes([0xC0 | n >> 6, 0x80 | n & 0x3F])
    return bytes([0xE0 | n >> 12, 0x80 | n >> 6 & 0x3F, 0x80 | n & 0x3F])


def flac_silence(frames: int, rate: int = 8000, channels: int = 1,
                 declare_length: bool = True) -> bytes:
    """A FLAC file of ``frames`` blocks of 65,535 silent samples per
    channel, each block about 14 bytes: 1 KiB decodes to about 10 MiB."""
    block = 65535
    total = frames * block if declare_length else 0
    bits = (rate << 44) | ((channels - 1) << 41) | (15 << 36) | total
    info = (struct.pack(">HH", block, block) + b"\0" * 6 + bits.to_bytes(8, "big")
            + b"\0" * 16)
    out = bytearray(b"fLaC" + bytes([0x80, 0, 0, 34]) + info)
    for n in range(frames):
        head = (bytes([0xFF, 0xF8, 0x70, (channels - 1) << 4]) + _utf8_number(n)
                + struct.pack(">H", block - 1))
        head += bytes([_crc8(head)])
        frame = head + b"\0\0\0" * channels          # one constant subframe each
        out += frame + struct.pack(">H", _crc16(frame))
    return bytes(out)


def wav(samples: np.ndarray, rate: int, channels: int = 1) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


def png_one_color(width: int, height: int) -> bytes:
    """A grayscale PNG of one color, built row by row, so the test never
    holds the decoded image."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    z = zlib.compressobj(9)
    row = b"\0" * (width + 1)
    idat = b"".join(z.compress(row) for _ in range(height)) + z.flush()
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _uri(data: bytes, kind: str = "image/png") -> str:
    return f"data:{kind};base64," + base64.b64encode(data).decode()


@pytest.fixture
def installed(monkeypatch):
    """The media checks over the real decoders."""
    monkeypatch.setattr(mg, "_allow_urls", False)
    media_sinks.install()
    yield
    media_sinks.uninstall()


# Audio

@pytest.mark.parametrize("declare_length", [True, False])
def test_flac_silence_is_refused_before_a_large_decode(installed, monkeypatch,
                                                       declare_length):
    monkeypatch.setattr(mg, "AUDIO_MAX_SAMPLES", 1 << 20)
    # 100 blocks: 1.3 KiB that decodes to 6.5 million samples.
    data = flac_silence(100, declare_length=declare_length)
    assert len(data) < 2048
    tracemalloc.start()
    try:
        with pytest.raises(MediaRefused) as err:
            _AUDIO_IO.read(io.BytesIO(data))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert str(err.value) == (
        "audio decodes to more than 1,048,576 samples, the limit for one audio clip, "
        "which is 2 minutes at 8,000 Hz with 1 channel. Send a shorter clip, or split "
        "the recording.")
    # The decode stopped near the limit: 2 MiB of samples, not 13 MiB.
    assert peak < 6 << 20


def test_chat_audio_refusal_is_a_400_of_the_generation_routes(installed, monkeypatch):
    from mlx_vlm.server.generation import PromptTooLongError

    monkeypatch.setattr(mg, "AUDIO_MAX_SAMPLES", 1 << 20)

    class Processor:
        def __call__(self, **kwargs):
            raise AssertionError("the processor ran for a refused clip")
    with pytest.raises(PromptTooLongError, match="the limit for one audio clip"):
        _UTILS.prepare_inputs(Processor(), audio=[io.BytesIO(flac_silence(100))],
                              prompts="hi")


def test_the_reader_gives_what_mlx_audio_gives(installed):
    stock = media_sinks._originals[(_AUDIO_IO, "read")]
    rng = np.random.default_rng(0)
    stereo = wav(rng.integers(-3000, 3000, 2000), 8000, channels=2)
    for data in (stereo, flac_silence(2), wav(np.arange(500), 16000)):
        for kwargs in ({}, {"dtype": "float32"}, {"dtype": "int16", "always_2d": True}):
            got, rate = md.read_audio(io.BytesIO(data), **kwargs)
            want, want_rate = stock(io.BytesIO(data), **kwargs)
            assert rate == want_rate and got.dtype == want.dtype
            assert np.array_equal(got, want)


def test_resampling_past_the_limit_is_refused_before_the_resample(installed,
                                                                  monkeypatch):
    import mlx_audio.utils as audio_utils

    monkeypatch.setattr(mg, "AUDIO_MAX_SAMPLES", 3000)
    monkeypatch.setattr(audio_utils, "resample_audio",
                        lambda *a, **k: pytest.fail("the resample ran"))
    clip = wav(np.zeros(2000), 1000)              # 2,000 samples at 1 kHz
    with pytest.raises(MediaRefused, match="at 16,000 Hz with 1 channel"):
        _UTILS.load_audio(clip, sr=16000)          # 32,000 samples at 16 kHz


def test_a_sample_rate_over_the_limit_is_refused(installed):
    with pytest.raises(MediaRefused, match="a sample rate of 400,000 Hz"):
        _AUDIO_IO.read(io.BytesIO(flac_silence(1, rate=400_000)))


@pytest.fixture
def endless_ffmpeg(tmp_path, monkeypatch):
    """An ffprobe that reports 48 kHz stereo, and an ffmpeg that writes
    zeros until it is stopped. The second gives the folder of a pid file
    the ffmpeg writes."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "ffprobe").write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        """echo '{"streams": [{"sample_rate": "48000", "channels": 2}]}'\n""")
    (tools / "ffmpeg").write_text(
        f"#!/bin/sh\necho $$ > {tmp_path}/ffmpeg.pid\nexec cat /dev/zero\n")
    for name in ("ffprobe", "ffmpeg"):
        (tools / name).chmod(0o755)
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    monkeypatch.setattr(settings, "SYSTEM_PATH", str(fixed))
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
    monkeypatch.setattr(mg, "AUDIO_MAX_SAMPLES", 1 << 18)
    return tmp_path


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_an_endless_ffmpeg_decode_is_stopped_at_the_limit(endless_ffmpeg):
    with pytest.raises(MediaRefused, match="the limit for one audio clip, which is "
                                           "2 seconds at 48,000 Hz with 2 channels"):
        media_programs.decode(b"OggS" + b"\0" * 64)
    assert _gone(int((endless_ffmpeg / "ffmpeg.pid").read_text()))
    with pytest.raises(MediaRefused, match="Split the recording"):
        media_programs.decode_mono(str(endless_ffmpeg / "a.ogg"), 16000)
    assert _gone(int((endless_ffmpeg / "ffmpeg.pid").read_text()))


def test_ffmpeg_gets_a_time_limit_past_the_sample_limit(tmp_path, monkeypatch):
    seen = []

    def run(argv, data, limit):
        seen.append((argv, limit))
        return 0, bytearray(b"\0\0"), b""
    monkeypatch.setattr(media_programs, "_run_bounded", run)
    monkeypatch.setattr(media_programs, "program", lambda name: name)
    monkeypatch.setattr(media_programs, "_run", lambda argv, data: type(
        "P", (), {"returncode": 0, "stderr": b"",
                  "stdout": b'{"streams": [{"sample_rate": "16000", "channels": 1}]}'})())
    monkeypatch.setattr(mg, "AUDIO_MAX_SAMPLES", 16000 * 60)
    media_programs.decode(b"x")
    media_programs.decode_mono("a.ogg", 16000)
    for argv, limit in seen:
        assert argv[argv.index("-t") + 1] == "61" and limit == 2 * 16000 * 60


def test_a_header_that_names_a_long_clip_is_refused_before_ffmpeg_runs(monkeypatch):
    monkeypatch.setattr(media_programs, "_run_bounded",
                        lambda *a: pytest.fail("ffmpeg ran"))
    monkeypatch.setattr(media_programs, "program", lambda name: name)
    probe = b'{"streams": [{"sample_rate": "16000", "channels": 1, "duration": "9000"}]}'
    monkeypatch.setattr(media_programs, "_run", lambda argv, data: type(
        "P", (), {"returncode": 0, "stderr": b"", "stdout": probe})())
    with pytest.raises(MediaRefused, match="2 hours 19 minutes at 16,000 Hz"):
        media_programs.decode(b"x")


def test_a_transcription_past_the_limit_is_a_400(monkeypatch):
    monkeypatch.setattr(stt, "import_mlx_whisper", lambda: object())
    monkeypatch.setattr(stt, "effective_model", lambda model, configured: configured)

    def decode_mono(path, rate, field="audio"):
        raise mg.audio_too_long(field, rate, 1, "Split the recording.")
    monkeypatch.setattr(media_programs, "decode_mono", decode_mono)
    with pytest.raises(stt.STTRequestError) as err:
        stt.run_transcription(b"x", filename="a.ogg", configured_model="m")
    assert err.value.status_code == 400
    assert str(err.value).startswith("the uploaded file decodes to more than 134,217,728 "
                                     "samples, the limit for one audio clip, which is 2 "
                                     "hours 19 minutes at 16,000 Hz with 1 channel.")


# Images

class _Stop(Exception):
    pass


class _Processor:
    """Stops prepare_inputs once every image has loaded."""

    def __call__(self, **kwargs):
        raise _Stop


class _StockLoads(list):
    """The size of each image that reached the stock loader, which would
    decode it. ``hook`` runs in the loader in place of the decode."""

    def __init__(self):
        super().__init__()
        self.hook = lambda: None

    def __call__(self, image, timeout=10):
        self.append(image.size)
        self.hook()
        return image


@pytest.fixture
def stock_loads(installed):
    loads = _StockLoads()
    checked = _UTILS.load_image
    # Set and put back by hand, so that uninstall then restores the stock
    # loader.
    _UTILS.load_image = media_sinks._image_loader(loads)
    yield loads
    _UTILS.load_image = checked


def test_many_large_one_color_pngs_are_refused_before_they_decode(stock_loads,
                                                                  monkeypatch):
    from mlx_vlm.server.generation import PromptTooLongError

    side = 2048
    monkeypatch.setattr(mg, "MEDIA_MAX_TOTAL_PIXELS", 4 * side * side)
    image = _uri(png_one_color(side, side))
    assert len(image) < 16 << 10                     # 16 KiB for 4 megapixels
    with pytest.raises(PromptTooLongError) as err:
        _UTILS.prepare_inputs(_Processor(), images=[image] * 5, prompts="hi")
    assert isinstance(err.value, MediaRefused)
    assert str(err.value) == (
        "image of 2048x2048 pixels brings the images and video frames of this request "
        "to 20,971,520 decoded pixels, over the limit of 16,777,216 pixels for one "
        "request. Send fewer or smaller images, or a shorter or smaller video.")
    # The fifth image was refused from its header, before the stock decode.
    assert stock_loads == [(side, side)] * 4


def test_each_request_gets_its_own_count(stock_loads, monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_TOTAL_PIXELS", 2 * 64 * 64)
    image = _uri(png_one_color(64, 64))
    for _ in range(3):
        with pytest.raises(ValueError, match="Failed to process inputs"):
            _UTILS.prepare_inputs(_Processor(), images=[image] * 2, prompts="hi")
    assert len(stock_loads) == 6


def test_too_many_images_are_refused(stock_loads, monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_ITEMS", 3)
    image = _uri(png_one_color(2, 2))
    with pytest.raises(MediaRefused, match="this request holds 4 or more images, audio "
                                           "clips and videos, over the limit of 3"):
        _UTILS.prepare_inputs(_Processor(), images=[image] * 4, prompts="hi")
    assert len(stock_loads) == 3


def test_one_request_at_a_time_decodes_media(stock_loads):
    inside, release = threading.Event(), threading.Event()
    order = []

    def hook():
        order.append(threading.current_thread().name)
        if threading.current_thread().name == "first":
            inside.set()
            release.wait()
    stock_loads.hook = hook
    image = _uri(png_one_color(2, 2))

    def run():
        with pytest.raises(ValueError):
            _UTILS.prepare_inputs(_Processor(), images=[image], prompts="hi")
    first = threading.Thread(target=run, name="first")
    second = threading.Thread(target=run, name="second")
    first.start()
    inside.wait()
    second.start()
    # The second request waits for the lock while the first one decodes.
    assert not md._DECODE_LOCK.acquire(blocking=False)
    assert order == ["first"]
    release.set()
    first.join()
    second.join()
    assert order == ["first", "second"]


def test_the_gate_counts_inline_images_from_their_headers(monkeypatch):
    app = FastAPI()

    @app.post("/v1/messages")
    async def route():
        return {"ok": True}
    mg.install_media_gate(False, app=app)
    client = TestClient(app)
    monkeypatch.setattr(mg, "MEDIA_MAX_TOTAL_PIXELS", 3 * 1024 * 1024)
    png = png_one_color(1024, 1024)
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}
    part = {"type": "image_url", "image_url": {"url": _uri(png)}}
    ok = {"messages": [{"role": "user", "content": [block, part, block]}]}
    assert client.post("/v1/messages", json=ok).status_code == 200
    over = {"messages": [{"role": "user", "content": [block, part, block, part]}]}
    r = client.post("/v1/messages", json=over)
    assert r.status_code == 400
    assert "over the limit of 3,145,728 pixels for one request" in r.text
    monkeypatch.setattr(mg, "MEDIA_MAX_ITEMS", 2)
    r = client.post("/v1/messages", json=ok)
    assert r.status_code == 400 and "over the limit of 2 for one request" in r.text


def test_the_gate_refuses_one_image_over_the_limit_from_its_header():
    with pytest.raises(MediaRefused, match="9000x8000 pixels, over the limit of "
                                           "67,108,864 pixels for one image"):
        mg.check_body({"type": "input_image",
                       "image_url": _uri(png_one_color(9000, 8000))},
                      "/v1/responses")


def test_image_edits_count_their_reference_images(monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_TOTAL_PIXELS", 2 * 64 * 64)
    image = _uri(png_one_color(64, 64))
    with mg._inline_image_files([image, image]) as paths:
        assert len(paths) == 2
    with pytest.raises(MediaRefused, match="image 3 brings"), \
            mg._inline_image_files([image] * 3):
        pass


# Video

def _video(path, frames: int, size=(64, 48)) -> str:
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    rng = np.random.default_rng(0)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return str(path)


def test_the_video_reader_gives_what_mlx_vlm_gives(tmp_path):
    path = _video(tmp_path / "v.avi", 40)
    stock = importlib.import_module("mlx_vlm.utils").load_video
    stock = getattr(stock, "__wrapped__", stock)
    for kwargs in ({}, {"fps": 5.0, "max_frames": 8}, {"nframes": 6}):
        got, fps = md.read_video(path, **kwargs)
        want, want_fps = stock(path, **kwargs)
        assert fps == want_fps and np.array_equal(got, want)


def test_a_video_past_the_pixel_limit_is_refused_before_a_frame_decodes(tmp_path,
                                                                       monkeypatch):
    import cv2
    path = _video(tmp_path / "v.avi", 40)
    monkeypatch.setattr(mg, "MEDIA_MAX_TOTAL_PIXELS", 7 * 64 * 48)
    read = []
    capture = cv2.VideoCapture

    class Capture:
        def __init__(self, p):
            self.cap = capture(p)

        def __getattr__(self, name):
            return getattr(self.cap, name)

        def read(self):
            read.append(1)
            return self.cap.read()
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    with pytest.raises(MediaRefused, match="video of 8 frames of 64x48 pixels brings"):
        md.read_video(path)
    assert read == []


def test_a_refused_video_in_the_frame_fallback_is_a_400(installed, monkeypatch):
    video = importlib.import_module("mlx_vlm.generate.video")

    def read_video(*args, **kwargs):
        raise mg.too_many_request_pixels("video of 600 frames of 1920x1080 pixels", 1)
    monkeypatch.setattr(md, "read_video", read_video)
    mp4 = _uri(b"\x00\x00\x00\x18ftypmp42" + b"\0" * 16, "video/mp4")
    with pytest.raises(HTTPException) as err:
        video.resolve_video_inputs(object(), [mp4])
    assert err.value.status_code == 400 and "600 frames" in err.value.detail


def test_a_stereo_clip_is_mixed_down_then_resampled(installed):
    from mlx_audio.utils import resample_audio

    rng = np.random.default_rng(1)
    left, right = rng.integers(-3000, 3000, 4800), rng.integers(-3000, 3000, 4800)
    clip = wav(np.stack([left, right], axis=1).reshape(-1), 48000, channels=2)
    got = _UTILS.load_audio(clip, sr=16000)
    mono = (np.stack([left, right], axis=1).astype(np.float32) / 32768.0).mean(axis=1)
    assert got.dtype == np.float32 and got.shape == (1600,)
    assert np.array_equal(got, resample_audio(mono, 48000, 16000))
    # A mono clip gives what mlx-vlm gives.
    stock = media_sinks._originals[(_UTILS, "load_audio")]
    one = wav(left, 48000)
    assert np.array_equal(_UTILS.load_audio(one, sr=16000), stock(io.BytesIO(one), 16000))
