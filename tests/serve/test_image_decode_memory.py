"""What an image's decode holds, measured on this Mac. The process's
memory counters read differently on CI's shared VM, so the module skips
there (KQUANT_FORCE_CPU) and the GPU gate runs it."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from PIL import Image

from gmlx.serve import image_service as svc

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="reads this Mac's memory counters")


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGB"), ("WEBP", "RGBA")])
def test_the_decode_price_bounds_what_the_decode_holds(fmt, mode, tmp_path):
    """What an image's decode holds at its peak, by the process's lifetime
    maximum physical footprint, stays within the price read from its
    header: 4096 pixels square, a PNG of 22 bytes a pixel and a WebP with
    alpha, the widest decode, of 35."""
    p = tmp_path / f"a.{fmt.lower()}"
    Image.new(mode, (4096, 4096), (10, 20, 30, 40)[:len(mode)]).save(p, fmt)
    price = svc.decode_price(p.read_bytes())[0]
    code = f"""
import ctypes, io, os, resource, sys
from PIL import Image
from gmlx.serve.image_service import decode_image
def peak():
    if sys.platform != "darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    # rusage_info_v4: ri_lifetime_max_phys_footprint at bytes 240:248.
    buf = ctypes.create_string_buffer(512)
    assert ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, buf) == 0
    return int.from_bytes(buf.raw[240:248], "little")
tiny = io.BytesIO()
Image.new({mode!r}, (2, 2)).save(tiny, {fmt!r})
decode_image(tiny.getvalue(), "image")
data = open({str(p)!r}, "rb").read()
before = peak()
decode_image(data, "image")
print(peak() - before)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=os.environ, check=True)
    held = int(out.stdout.split()[-1])
    assert 0.8 * price < held <= price, (held, price)
