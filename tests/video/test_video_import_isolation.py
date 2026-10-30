"""Importing gmlx, gmlx.image, the image serve paths or the CLI never
imports gmlx.video, and identifying a video file or importing the video
encoder module or the VAE schedules loads no mlx. Checked in a child
process so earlier tests cannot mask it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def test_the_image_and_serve_paths_leave_gmlx_video_unloaded():
    code = (
        "import sys\n"
        "import gmlx\n"
        "import gmlx.image.fingerprint, gmlx.image.loader\n"
        "import gmlx.serve.image_service, gmlx.serve.image_steps\n"
        "import gmlx.serve.server, gmlx.serve.patches.image_steps\n"
        "import gmlx.commands.cli, gmlx.commands.image\n"
        "bad = sorted(m for m in sys.modules if m.startswith('gmlx.video'))\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_identifying_a_video_file_loads_no_mlx():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'tests/video')\n"
        "from video_fixtures import H3_HEADER, fixture_view\n"
        "from gmlx.video import fingerprint as vfp\n"
        "import gmlx.video.encoder, gmlx.video.vae.chunking\n"
        "assert vfp.identify(fixture_view(H3_HEADER)).family.name == 'h3'\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'mlx')\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
