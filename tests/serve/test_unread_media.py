"""Media an LFM3 model has no tower for gets a 400 before its processor
runs. Other families answer as before."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gmlx._vendor.mlx_vlm.server import openai as route

IMAGE, VIDEO, CLIP = ["a.png"], ["a.mp4"], [b"RIFF"]


def _model(model_type, *, vision=True, audio=True):
    return SimpleNamespace(config=SimpleNamespace(
        model_type=model_type,
        vision_config=object() if vision else None,
        audio_config=object() if audio else None))


@pytest.mark.parametrize("images, videos", [(IMAGE, ()), ((), VIDEO)])
def test_lfm3_audio_refuses_an_image_or_a_video(images, videos):
    with pytest.raises(HTTPException) as e:
        route._refuse_unread_media(_model("lfm3_audio", vision=False),
                                   images, videos, CLIP)
    assert e.value.status_code == 400
    assert e.value.detail == "this model takes no image or video input"


def test_lfm3_vl_refuses_a_clip():
    with pytest.raises(HTTPException) as e:
        route._refuse_unread_media(_model("lfm3_vl", audio=False), IMAGE,
                                   VIDEO, CLIP)
    assert e.value.detail == "this model takes no audio input"


def test_what_a_model_reads_passes():
    route._refuse_unread_media(_model("lfm3_omni"), IMAGE, VIDEO, CLIP)
    route._refuse_unread_media(_model("lfm3_audio", vision=False), (), (), CLIP)
    route._refuse_unread_media(_model("lfm3_vl", audio=False), IMAGE, VIDEO, ())


@pytest.mark.parametrize("model_type", ["gemma3", "nemotron_h_transcribe",
                                        "qwen3_5", "lfm3"])
def test_other_families_answer_as_before(model_type):
    """The rule reads only the LFM3 container types, so another family's
    request still reaches its own processor."""
    route._refuse_unread_media(_model(model_type, vision=False, audio=False),
                               IMAGE, VIDEO, CLIP)
    route._refuse_unread_media(SimpleNamespace(), IMAGE, VIDEO, CLIP)
