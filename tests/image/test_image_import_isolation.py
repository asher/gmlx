"""Importing gmlx, or loading a text model's loader, never imports
gmlx.image. Checked in a child process so earlier tests cannot mask it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def test_import_gmlx_leaves_gmlx_image_unloaded():
    code = (
        "import sys\n"
        "import gmlx\n"
        "gmlx.load_model\n"
        "import gmlx.load.discovery\n"
        "bad = sorted(m for m in sys.modules if m.startswith('gmlx.image'))\n"
        "assert not bad, bad\n"
        "import gmlx.image.registry\n"
        "assert 'gmlx.image' in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
