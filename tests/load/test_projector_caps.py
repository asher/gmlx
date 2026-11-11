"""What a projector's header says it gives its text model
(``discovery.projector_caps``)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gmlx.load.discovery as disc  # noqa: E402
from gmlx.load.discovery import Caps, projector_caps  # noqa: E402
from lfm3_header import write_projector_header  # noqa: E402


def test_the_caps_come_from_the_header_keys(tmp_path):
    omni = write_projector_header(tmp_path / "omni.gguf", vision=True,
                                  audio=True, speech=True)
    eyes = write_projector_header(tmp_path / "eyes.gguf", vision=True)
    ears = write_projector_header(tmp_path / "ears.gguf", audio=True)
    assert projector_caps(omni) == Caps(vision=True, audio_in=True,
                                        speech_out=True)
    assert projector_caps(eyes) == Caps(vision=True)
    assert projector_caps(ears) == Caps(audio_in=True)


def test_a_nemotron_speech_encoder_is_named_by_its_arch(tmp_path):
    pee = write_projector_header(tmp_path / "encoder.gguf", arch="pee")
    assert projector_caps(pee) == Caps(speech_encoder=True)
    assert not projector_caps(
        write_projector_header(tmp_path / "p.gguf", audio=True)).speech_encoder


def test_no_file_and_no_gguf_have_no_caps(tmp_path):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf")
    for path in (None, "", tmp_path / "absent.gguf", junk):
        assert projector_caps(path) == Caps()


def test_a_rewritten_file_is_read_again(tmp_path):
    path = tmp_path / "p.gguf"
    write_projector_header(path, vision=True)
    assert projector_caps(path) == Caps(vision=True)
    write_projector_header(path, vision=True, audio=True)
    os.utime(path, ns=(1, 1))
    assert projector_caps(path) == Caps(vision=True, audio_in=True)


def test_one_file_is_scanned_once(tmp_path, monkeypatch):
    from gmlx.load import headerscan

    path = write_projector_header(tmp_path / "p.gguf", audio=True)
    scans = []
    real = headerscan.scan_gguf

    def scan(p, **kw):
        scans.append(p)
        return real(p, **kw)

    monkeypatch.setattr(headerscan, "scan_gguf", scan)
    monkeypatch.setattr(disc, "_CAPS_MEMO", {})
    assert projector_caps(path) == projector_caps(path)
    assert len(scans) == 1
