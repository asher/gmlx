"""gmlx/serve/media_sinks.py: the functions that open request media or load
by path check the reference themselves, whatever route or body shape
carried it."""

from __future__ import annotations

import base64
import importlib
import io
import os

import numpy as np
import pytest
from PIL import Image

from gmlx.serve import media_sinks as ms
from gmlx.serve.patches import media_gate as mg
from gmlx.serve.patches.media_gate import MediaRefused

_UTILS = importlib.import_module("mlx_vlm.utils")
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16


def _png(size=(2, 2), fmt="PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size).save(out, format=fmt)
    return out.getvalue()


def _uri(data: bytes, kind="image/png") -> str:
    return f"data:{kind};base64," + base64.b64encode(data).decode()


class _Recorder:
    """Stands in for one stock function and records what reached it."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs) if callable(self.result) else self.result


@pytest.fixture
def sinks(monkeypatch, tmp_path):
    """Install the checks over recorders, with a media folder in tmp_path.
    Yields the recorders by name."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    rec = {name: _Recorder() for name in ("load_image", "load_audio", "load_video",
                                          "load", "load_drafter")}
    rec["load_image"].result = lambda image, timeout=10: image
    for name in ("load_image", "load_audio", "load_video", "load"):
        monkeypatch.setattr(_UTILS, name, rec[name])
    monkeypatch.setattr(gen, "load", rec["load"])
    monkeypatch.setattr(drafters, "load_drafter", rec["load_drafter"])
    monkeypatch.setattr(mg, "_allow_urls", False)
    root = ms.ensure_media_root(str(tmp_path / "cache" / "gmlx" / "media"))
    ms.install()
    rec["root"] = root
    yield rec
    ms.uninstall()
    mg.set_media_root(None)


# The media folder

def test_the_media_folder_is_created_private(tmp_path):
    root = str(tmp_path / "cache" / "gmlx" / "media")
    try:
        assert ms.ensure_media_root(root) == root
        assert os.stat(root).st_mode & 0o777 == 0o700
        assert mg.media_root() == root
    finally:
        mg.set_media_root(None)


def test_a_media_folder_open_to_others_is_made_private(tmp_path):
    root = tmp_path / "media"
    root.mkdir(mode=0o777)
    os.chmod(root, 0o777)
    try:
        assert ms.ensure_media_root(str(root)) == str(root)
        assert os.stat(root).st_mode & 0o777 == 0o700
    finally:
        mg.set_media_root(None)


def test_the_media_folder_matches_the_form_macos_gives_it(sinks, tmp_path):
    # A cache folder reached through a link, as /var is on macOS: a request
    # may name a file through the link or by the folder's real path.
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    root = ms.ensure_media_root(str(tmp_path / "link" / "media"))
    with open(tmp_path / "real" / "media" / "a.png", "wb") as f:
        f.write(_png((5, 6)))
    for folder in (tmp_path / "link" / "media", tmp_path / "real" / "media"):
        assert _UTILS.load_image(str(folder / "a.png")).size == (5, 6)
    assert root == str(tmp_path / "link" / "media")


def test_a_media_folder_that_is_a_link_is_not_used(tmp_path, caplog):
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "media")
    assert ms.ensure_media_root(str(tmp_path / "media")) is None
    assert mg.media_root() is None and mg.media_parts(str(tmp_path / "media/a.png")) is None
    assert "symbolic link" in caplog.text


# Images

def test_an_inline_image_reaches_the_loader_decoded_lazily(sinks):
    out = _UTILS.load_image(_uri(_png()))
    ((image, _), _), = sinks["load_image"].calls
    assert isinstance(image, Image.Image) and out is image and image.format == "PNG"


@pytest.mark.parametrize("value", [
    "/Users/me/.ssh/id_ed25519", "file:///etc/passwd", "~/Desktop/a.png", "a.png",
    "../a.png"])
def test_an_image_path_outside_the_media_folder_is_refused(sinks, value):
    with pytest.raises(MediaRefused) as e:
        _UTILS.load_image(value)
    assert f"cp -c FILE {sinks['root']}/" in str(e.value)
    assert sinks["load_image"].calls == []


def test_an_image_in_the_media_folder_is_read(sinks):
    path = os.path.join(sinks["root"], "shots", "a.png")
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(_png((3, 4)))
    for ref in (path, "file://" + path):
        image = _UTILS.load_image(ref)
        assert image.size == (3, 4)


def test_a_link_in_the_media_folder_is_not_followed(sinks, tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(_png())
    os.symlink(secret, os.path.join(sinks["root"], "a.png"))
    os.symlink(tmp_path, os.path.join(sinks["root"], "dir"))
    for name in ("a.png", "dir/secret.png"):
        with pytest.raises(MediaRefused, match="symbolic link"):
            _UTILS.load_image(os.path.join(sinks["root"], name))
    with pytest.raises(MediaRefused):
        _UTILS.load_image(os.path.join(sinks["root"], "..", "secret.png"))
    assert sinks["load_image"].calls == []


def test_a_media_folder_path_names_a_regular_file_only(sinks):
    os.mkfifo(os.path.join(sinks["root"], "pipe"))
    with pytest.raises(MediaRefused, match="not a regular file"):
        _UTILS.load_image(os.path.join(sinks["root"], "pipe"))


def test_a_url_is_refused_unless_media_urls_is_on(sinks, monkeypatch):
    with pytest.raises(MediaRefused, match="server.media_urls"):
        _UTILS.load_image("https://example.com/a.png")
    import gmlx.serve.media_fetch as mf
    fetched = []
    monkeypatch.setattr(mf, "fetch", lambda url, *, max_bytes: fetched.append(
        (url, max_bytes)) or _png())
    monkeypatch.setattr(mg, "_allow_urls", True)
    assert _UTILS.load_image("https://example.com/a.png").size == (2, 2)
    assert fetched == [("https://example.com/a.png", mg.MEDIA_MAX_BYTES)]


def test_a_refused_fetch_is_a_media_refusal(sinks, monkeypatch):
    import gmlx.serve.media_fetch as mf

    def refuse(url, *, max_bytes):
        raise mf.FetchRefused("the host is on this Mac")
    monkeypatch.setattr(mf, "fetch", refuse)
    monkeypatch.setattr(mg, "_allow_urls", True)
    with pytest.raises(MediaRefused, match="the host is on this Mac"):
        _UTILS.load_image("http://127.0.0.1/a.png")


def test_image_limits_apply_to_every_image(sinks, monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_PIXELS", 63)
    with pytest.raises(MediaRefused, match="8x8 pixels"):
        _UTILS.load_image(_uri(_png((8, 8))))
    with pytest.raises(MediaRefused, match="8x8 pixels"):
        _UTILS.load_image(Image.new("RGB", (8, 8)))
    monkeypatch.setattr(mg, "MEDIA_MAX_BYTES", 16)
    with pytest.raises(MediaRefused, match="larger than"):
        _UTILS.load_image(io.BytesIO(_png()))
    with pytest.raises(MediaRefused, match="larger than"):
        _UTILS.load_image(_uri(_png()))
    assert sinks["load_image"].calls == []


def test_an_image_over_pillows_own_limit_gets_the_pixel_message(sinks, monkeypatch):
    # Pillow refuses to open an image over twice its limit, before the size
    # check here can run.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(MediaRefused, match="over the limit of"):
        _UTILS.load_image(_uri(_png((8, 8))))
    with pytest.raises(MediaRefused, match="over the limit of"):
        mg._decode_edit_image(_uri(_png((8, 8))))
    assert sinks["load_image"].calls == []


def test_another_scheme_is_named_in_the_sink_refusal(sinks, monkeypatch):
    with pytest.raises(MediaRefused, match="names a ftp:// URL"):
        _UTILS.load_image("ftp://host/a.png")


def test_only_the_common_image_formats_are_read(sinks):
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n"
    with pytest.raises(MediaRefused, match="not a readable"):
        _UTILS.load_image(_uri(eps, "image/eps"))
    assert _UTILS.load_image(_uri(_png(fmt="BMP"), "image/bmp")).format == "BMP"


def test_modules_that_copied_the_image_loader_get_the_check(sinks):
    rerank = importlib.import_module("mlx_vlm.server.reranking")
    embeddings = importlib.import_module("mlx_vlm.server.embeddings")
    assert rerank.load_image is _UTILS.load_image
    assert rerank.load_video is _UTILS.load_video
    assert embeddings.load_image is _UTILS.load_image
    # A rerank document that names a key file, which the body walk missed.
    with pytest.raises(MediaRefused):
        rerank.load_image("/Users/me/.ssh/id_ed25519")


# Audio

def test_audio_arrives_inline_or_from_the_media_folder(sinks):
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
    samples = np.zeros(4, dtype=np.float32)
    _UTILS.load_audio(samples, 16000)
    _UTILS.load_audio(io.BytesIO(wav), 16000)
    _UTILS.load_audio(_uri(wav, "audio/wav"), 16000)
    path = os.path.join(sinks["root"], "a.wav")
    with open(path, "wb") as f:
        f.write(wav)
    _UTILS.load_audio(path, 16000)
    got = [args[0] for args, _ in sinks["load_audio"].calls]
    assert got[0] is samples
    assert [g.getvalue() for g in got[1:]] == [wav, wav, wav]


@pytest.mark.parametrize("value", ["/Users/me/a.wav", "file:///etc/passwd",
                                   "http://example.com/a.wav"])
def test_an_audio_reference_the_server_does_not_open_is_refused(sinks, value):
    with pytest.raises(MediaRefused):
        _UTILS.load_audio(value, 16000)
    assert sinks["load_audio"].calls == []


def test_a_media_folder_file_over_the_limit_is_refused(sinks, monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_BYTES", 16)
    path = os.path.join(sinks["root"], "big.wav")
    with open(path, "wb") as f:
        f.write(b"x" * 17)
    with pytest.raises(MediaRefused, match="larger than"):
        _UTILS.load_audio(path, 16000)
    with pytest.raises(MediaRefused, match="larger than"):
        _UTILS.load_image(path)
    assert sinks["load_audio"].calls == [] and sinks["load_image"].calls == []
    # A video there is streamed from the file, so its size is not limited.
    with open(path, "wb") as f:
        f.write(MP4 + b"x" * 64)
    _UTILS.load_video(path)
    assert sinks["load_video"].calls[0][0][0] == path


def test_inline_audio_over_the_limit_is_refused(sinks, monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_BYTES", 4)
    with pytest.raises(MediaRefused, match="larger than"):
        _UTILS.load_audio(io.BytesIO(b"RIFF\x24\x00"), 16000)


# Video

def test_an_inline_video_reaches_the_reader_as_a_private_file(sinks):
    seen = []

    def reader(path, *a, **k):
        seen.append((path, os.path.exists(path), oct(os.stat(path).st_mode & 0o777),
                     open(path, "rb").read()))
        return "frames"
    sinks["load_video"].result = reader
    assert _UTILS.load_video(_uri(MP4, "video/mp4"), fps=2.0) == "frames"
    (path, existed, mode, data), = seen
    assert existed and mode == "0o600" and data == MP4
    assert not os.path.exists(path)


def test_a_video_in_the_media_folder_is_read_where_it_is(sinks):
    path = os.path.join(sinks["root"], "clip.mp4")
    with open(path, "wb") as f:
        f.write(MP4)
    _UTILS.load_video(path)
    assert sinks["load_video"].calls[0][0][0] == path


@pytest.mark.parametrize("data", [b"#EXTM3U\n/Users/me/secret.mp4\n", b"hello"])
def test_a_video_that_is_not_a_video_container_is_refused(sinks, data):
    path = os.path.join(sinks["root"], "list.m3u8")
    with open(path, "wb") as f:
        f.write(data)
    for ref in (path, _uri(data, "video/mp4")):
        with pytest.raises(MediaRefused, match="no MP4"):
            _UTILS.load_video(ref)
    assert sinks["load_video"].calls == []


def test_a_video_path_outside_the_media_folder_is_refused(sinks):
    with pytest.raises(MediaRefused):
        _UTILS.load_video("/Users/me/movie.mp4")
    assert sinks["load_video"].calls == []


# Model loads and the APC disk tier

def test_only_a_configured_model_loads(sinks, tmp_path):
    import gmlx.serve.bridge_vlm as serving
    from gmlx.config import build_config
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    serving.clear_resolved_models()
    serving.register_resolved_models(build_config({"models": {"m": {"path": str(model)}}}))
    try:
        gen = importlib.import_module("mlx_vlm.server.generation")
        assert gen.load is _UTILS.load
        _UTILS.load(str(model))
        assert len(sinks["load"].calls) == 1
        for path in ("/Users/me/Library", str(tmp_path / "other")):
            with pytest.raises(ms.ModelRefused, match="not a configured model"):
                gen.load(path)
        with pytest.raises(ms.ModelRefused, match="not a configured adapter"):
            _UTILS.load(str(model), str(tmp_path / "adapter"))
        assert len(sinks["load"].calls) == 1
    finally:
        serving.clear_resolved_models()


def test_only_the_drafter_the_build_chose_loads(sinks, monkeypatch):
    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    with pytest.raises(ms.ModelRefused):
        drafters.load_drafter("/tmp/drafter")
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "/models/m.gguf")
    with pytest.raises(ms.ModelRefused):
        drafters.load_drafter("/tmp/drafter")
    drafters.load_drafter("/models/m.gguf", kind="mtp")
    assert sinks["load_drafter"].calls == [(("/models/m.gguf",), {"kind": "mtp"})]


def test_the_apc_disk_tier_stores_only_where_the_build_chose(sinks, monkeypatch, tmp_path):
    apc = importlib.import_module("mlx_vlm.apc")
    monkeypatch.setenv("APC_DISK_PATH", str(tmp_path / "apc"))
    with pytest.raises(ms.ModelRefused):
        apc.DiskBlockStore(tmp_path / "chosen-by-request")
    assert not (tmp_path / "chosen-by-request").exists()
    store = apc.DiskBlockStore(tmp_path / "apc", namespace="ns")
    try:
        assert store.dir == tmp_path / "apc" / "ns"
    finally:
        store.close() if hasattr(store, "close") else None


def test_uninstall_restores_every_function(sinks):
    wrapped = _UTILS.load_image
    ms.uninstall()
    assert _UTILS.load_image is sinks["load_image"] and wrapped is not _UTILS.load_image
    rerank = importlib.import_module("mlx_vlm.server.reranking")
    assert rerank.load_image is not wrapped
    ms.install()
    assert _UTILS.load_image is not sinks["load_image"]
