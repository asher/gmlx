"""The survivor-read hang parks one thread; the interpreter lives.

Every buffer-read binding releases the GIL around its wait
(to_scalar, tolist, the nd-array export), so a thread stuck on an
unsignaled event does not freeze the process: other threads keep
running and an in-process watchdog thread can observe the hang and
kill the process itself. Supervision design rests on this
(a hung worker answers health checks; only the stalled thread's own
progress signal goes quiet).

Arms:
  hang - trip async, drain, ticker thread at 200 ms, then read the
         survivor on the main thread. Pre-0.32.1: the read never
         returns; the ticker keeps printing and after 20 post-entry
         ticks calls os._exit(42), proving watchdog viability.
         Post-fix: the read returns promptly and correct (rc 0).
  busy - control: same ticker, main thread runs real evals for 1.5 s
         wall clock instead of a blocked read. Ticks must interleave
         on every wheel; this validates the ticker mechanism the
         hang arm depends on.
"""

import os
import sys
import threading
import time

import mlx.core as mx

from _common import MID_VAL, check, survivor_and_trip

entered = threading.Event()


def ticker():
    post = 0
    while True:
        time.sleep(0.2)
        if entered.is_set():
            post += 1
            print(f"tick post={post}", flush=True)
            if post >= 20:
                print("outcome=ALIVE watchdog os._exit(42)", flush=True)
                os._exit(42)
        else:
            print("tick pre", flush=True)


def main():
    arm = sys.argv[1]

    s = mx.new_stream(mx.default_device())
    with mx.stream(s):
        mid, big = survivor_and_trip()
    try:
        mx.async_eval(big)
        print("outcome=NO-THROW", flush=True)
        return
    except Exception:
        pass
    mx.synchronize(s)

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    time.sleep(1.0)

    entered.set()
    print("entering read", flush=True)
    if arm == "hang":
        v = mid.tolist()[0]
        print(f"outcome={check(v, MID_VAL)} got={v:.6g}", flush=True)
    elif arm == "busy":
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            a = mx.random.normal((2048, 2048))
            mx.eval(a @ a)
        print("outcome=BUSY-DONE", flush=True)


if __name__ == "__main__":
    main()
