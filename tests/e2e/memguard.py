"""Kill the e2e servers before the box swaps itself into a watchdog panic.

A streaming server wires most of RAM. When the rest of the box then goes
to swap, the kernel has nothing left to reclaim and the watchdog panics.
Run this beside a live e2e run:

    python tests/e2e/memguard.py --verbose --log ~/.cache/gmlx/e2e/memguard.log

Samples vm_stat and swap twice a second. When the compressor grows past
``--comp-gb`` over its start value, or free RAM falls under
``--floor-gb`` with swap past ``--swap-gb``, it SIGKILLs every process
whose command line matches ``--pattern``. With ``--verbose`` it logs one
line per sample: wired, anonymous, file-backed, compressor and swap
bytes, and the four largest processes by RSS, which is the timeline a
panic leaves behind otherwise.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

PAGE = 16384
KEYS = ("free", "inactive", "speculative", "purgeable", "wired down",
        "File-backed", "Anonymous", "Pages stored in compressor")


def vm() -> dict:
    txt = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = {}
    for k in KEYS:
        m = re.search(rf"^(?:Pages )?{re.escape(k)}[^:]*:\s+(\d+)", txt,
                      re.M | re.I)
        g[k] = int(m.group(1)) * PAGE if m else 0
    return g


def swap_used() -> float:
    txt = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True).stdout
    m = re.search(r"used = ([\d.]+)M", txt)
    return float(m.group(1)) * 1e6 if m else 0.0


def top_rss(n: int = 4) -> str:
    ps = subprocess.run(["ps", "-axo", "pid=,rss=,comm="], capture_output=True,
                        text=True).stdout
    rows = []
    for line in ps.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[0]) != os.getpid():
            rows.append((int(parts[1]) * 1024, parts[0],
                         os.path.basename(parts[2])[:18]))
    rows.sort(reverse=True)
    return " ".join(f"{name}[{pid}]:{r / 1e9:.1f}" for r, pid, name in rows[:n])


def kill_all(pattern: str) -> list:
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                        text=True).stdout
    victims = []
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if re.search(pattern, cmd) and int(pid) != os.getpid():
            victims.append((int(pid), cmd[:80]))
    for pid, _ in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return victims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--floor-gb", type=float, default=3.0)
    ap.add_argument("--swap-gb", type=float, default=3.0)
    ap.add_argument("--comp-gb", type=float, default=12.0,
                    help="compressor growth over its start value that kills")
    ap.add_argument("--pattern", default=(r"gmlx serve|gmlx\.serve|-m gmlx serve"
                                          r"|run_server_e2e|run_stream_e2e"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    out = open(a.log, "a") if a.log else sys.stdout
    comp0 = vm()["Pages stored in compressor"]
    while True:
        g = vm()
        free = g["free"] + g["inactive"] + g["speculative"] + g["purgeable"]
        comp = g["Pages stored in compressor"]
        sw = swap_used()
        line = (f"{time.strftime('%H:%M:%S')} free {free / 1e9:6.1f} wired "
                f"{g['wired down'] / 1e9:6.1f} anon {g['Anonymous'] / 1e9:5.1f} "
                f"file {g['File-backed'] / 1e9:5.1f} compressor {comp / 1e9:5.1f} "
                f"swap {sw / 1e9:5.1f} GB top {top_rss()}")
        if a.verbose:
            print(line, file=out, flush=True)
        if ((free < a.floor_gb * 1e9 and sw > a.swap_gb * 1e9)
                or comp > comp0 + a.comp_gb * 1e9):
            print(f"TRIGGER {line} -> killed {kill_all(a.pattern)}", file=out,
                  flush=True)
            time.sleep(5)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
