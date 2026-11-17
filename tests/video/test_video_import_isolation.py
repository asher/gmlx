"""Importing gmlx, gmlx.image, the image serve paths or the CLI never
imports gmlx.video, and identifying a video file or a MiniMax-H3-Image
file or importing the video encoder module, the image resizes, the H3
picture prompt code, the anchor rule or the VAE schedules loads no mlx.
Checked in a child process so earlier tests cannot mask it."""
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
        "import gmlx.image.families.h3_image.family\n"
        "import gmlx.image.families.h3_image.model\n"
        "import gmlx.image.vae.h3_image, gmlx.image.h3.lora_map\n"
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
        "import gmlx.video.encoder, gmlx.video.images\n"
        "import gmlx.video.vae.chunking\n"
        "import gmlx.image.int8, gmlx.image.h3.layout\n"
        "import gmlx.image.h3.encoder, gmlx.image.h3.tiles\n"
        "import gmlx.image.h3.pictures, gmlx.video.families.h3.keyframes\n"
        "import gmlx.image.h3.anchors, gmlx.video.anchors\n"
        "assert vfp.identify(fixture_view(H3_HEADER)).family.name == 'h3'\n"
        "sys.path.insert(0, 'tests/image')\n"
        "import image_fixtures\n"
        "from gmlx.image import fingerprint as ifp\n"
        "view = image_fixtures.fixture_view("
        "'h3_image_pruned_ud_q4_k_xl.header.json')\n"
        "assert ifp.identify(view).family.name == 'h3_image'\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'mlx')\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
