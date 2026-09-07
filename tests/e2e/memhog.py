#!/usr/bin/env python3
"""Wire memory and hold it: a stand-in for another process on the box.

Usage: python tests/e2e/memhog.py --gb N [--hold SECONDS]
Prints ``wired <bytes> <mode>`` once the pages are locked, then holds
until SIGTERM or the hold time. Exit releases everything. ``mode`` is
``mlock``, or ``touch`` when mlock is refused (dirty anonymous pages
then take the same room from the reclaimable pool).
"""
from __future__ import annotations

import argparse
import ctypes
import mmap
import signal
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, required=True, help="GiB to wire")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="seconds to hold; 0 holds until SIGTERM")
    a = ap.parse_args()
    n = int(a.gb * (1 << 30))
    m = mmap.mmap(-1, n)
    buf = (ctypes.c_char * n).from_buffer(m)
    addr = ctypes.c_void_p(ctypes.addressof(buf))
    libc = ctypes.CDLL(None, use_errno=True)
    mode = "mlock"
    if libc.mlock(addr, ctypes.c_size_t(n)) != 0:
        mode = "touch"
        for off in range(0, n, mmap.PAGESIZE):
            buf[off] = b"\x01"
    print(f"wired {n} {mode}", flush=True)
    stop: list = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    deadline = time.monotonic() + a.hold if a.hold else None
    while not stop and (deadline is None or time.monotonic() < deadline):
        time.sleep(0.2)
    if mode == "mlock":
        libc.munlock(addr, ctypes.c_size_t(n))
    del buf
    m.close()
    print("released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
