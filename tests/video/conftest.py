"""Fixtures of the video tests: a stand-in ffmpeg written once per session
and linked into each test's folder, since macOS checks each new program
file the first time it runs."""

from __future__ import annotations

import os
import stat

import pytest

# What the stand-in prints for -encoders, unless a test writes another.
ENCODERS = """Encoders:
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder
 A....D aac                  AAC (Advanced Audio Coding)
"""

# The program: -encoders prints encoders.txt beside the link, any other
# call runs ffmpeg-body.sh beside it with $last the output path, its file:
# protocol dropped.
_PROGRAM = """#!/bin/sh
for last; do :; done
last="${last#file:}"
here=$(dirname "$0")
case " $* " in *" -encoders "*) cat "$here/encoders.txt"; exit 0;; esac
. "$here/ffmpeg-body.sh"
"""


@pytest.fixture(scope="session")
def _ffmpeg_program(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("programs") / "ffmpeg"
    path.write_text(_PROGRAM)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def stand_in_ffmpeg(_ffmpeg_program, tmp_path):
    """``make(body, encoders=ENCODERS)``: the session's stand-in ffmpeg
    linked into this test's folder, running the shell ``body`` with
    ``$last`` the output path and ``$here`` the folder. Returns its
    path."""
    def make(body: str = 'cat > "$last"', encoders: str = ENCODERS) -> str:
        (tmp_path / "ffmpeg-body.sh").write_text(body + "\n")
        (tmp_path / "encoders.txt").write_text(encoders)
        link = tmp_path / "ffmpeg"
        if not link.exists():
            os.symlink(_ffmpeg_program, link)
        return str(link)
    return make
