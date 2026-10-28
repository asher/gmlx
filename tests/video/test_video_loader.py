"""The video loader's handling of what the fingerprint found: a warning for
unknown weights or an unverified qkv layout, a refusal for a known layout
other than [q; k; v], and the image loader's refusals as video errors.
CPU-only; the image loader is stubbed."""
from __future__ import annotations

import dataclasses

import pytest
from video_fixtures import H3_HEADER, fixture_view

from gmlx.image import fingerprint as ifp
from gmlx.image import loader as image_loader
from gmlx.video import fingerprint as vfp
from gmlx.video import loader as video_loader


def _load(monkeypatch, **fields):
    """Run load_dit on the header fixture with the identity ``fields``
    replaced. Returns the log lines and the image loader's arguments."""
    vid = dataclasses.replace(vfp.identify(fixture_view(H3_HEADER)), **fields)
    monkeypatch.setattr(video_loader, "identify_dit", lambda *a, **k: vid)
    calls = []

    def stub(path, **kw):
        calls.append(kw)
        return "model", None

    monkeypatch.setattr(image_loader, "load_dit", stub)
    logs = []
    model, got = video_loader.load_dit("h3.gguf", log=logs.append)
    assert (model, got) == ("model", vid)
    return logs, calls


def test_a_known_file_loads_without_a_warning(monkeypatch):
    logs, calls = _load(monkeypatch, variant="fl2va", layout="qkv")
    assert logs == []
    assert calls[0]["kind"] == "video"
    assert calls[0]["identified"].layout == "qkv"


def test_an_unknown_file_loads_with_both_warnings(monkeypatch):
    logs, _ = _load(monkeypatch, variant="unknown", layout="unverified")
    assert logs == [
        "[video] warning: this MiniMax-H3 GGUF matches no known MiniMax-H3 "
        "weights; it runs as fl2va",
        "[video] warning: this MiniMax-H3 GGUF is not a known file, so its "
        "qkv row order is unverified; it runs as [q; k; v]"]


def test_a_known_per_head_layout_is_refused(monkeypatch):
    with pytest.raises(vfp.VideoModelError) as ei:
        _load(monkeypatch, variant="fl2va", layout="per_head")
    assert str(ei.value) == ("this MiniMax-H3 GGUF stores qkv as per_head, "
                             "which gmlx video does not run")


def test_an_image_loader_refusal_is_a_video_error(monkeypatch):
    vid = vfp.identify(fixture_view(H3_HEADER))
    monkeypatch.setattr(video_loader, "identify_dit", lambda *a, **k: vid)

    def refuse(path, **kw):
        raise ifp.ImageModelError("h3.gguf: truncated GGUF")

    monkeypatch.setattr(image_loader, "load_dit", refuse)
    with pytest.raises(vfp.VideoModelError, match="truncated GGUF"):
        video_loader.load_dit("h3.gguf", log=lambda *_: None)


def test_a_file_that_is_not_a_gguf_is_a_video_error(tmp_path):
    path = tmp_path / "h3.gguf"
    path.write_bytes(b"NOTAGGUF" + bytes(64))
    with pytest.raises(vfp.VideoModelError):
        video_loader.identify_dit(str(path))


def test_the_reshape_refusal_names_the_command():
    key = image_loader._ORIG_SHAPE_KEY + "blocks.0.mlp.fc1.weight"
    with pytest.raises(ifp.ImageModelError, match="gmlx video loads DiT"):
        image_loader._refuse_reshaped([key], "video")
