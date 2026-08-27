"""Async survivors after a clean drain: rehabilitated or landmine?

mx.async_eval attaches a per-stream event to each array at its tape
step and signals it only in the trailing loop after the tape
completes. Before mlx 0.32.1, a mid-tape throw skipped that loop, so
a survivor's event could never signal and never detach (detach
requires is_signaled). mlx 0.32.1 adds an unwind handler
(ml-explore/mlx#3675) that signals the events and synchronizes the
open streams before rethrowing.

Arms, one subprocess each:
  item       - buffer-read the survivor (tolist -> array::wait on the
               event). Pre-fix: permanent hang, killed externally.
               Post-fix: prompt, correct.
  xstream    - consume the survivor on another stream. Pre-fix: the
               tape encodes an unsatisfiable event wait, GPU timeout,
               abort. Post-fix: clean (signaled events detach).
  samestream - consume on the survivor's own stream: the wait branch
               is skipped. Clean on every wheel.
  fresh      - control: fresh graph on the stream. Clean everywhere.
"""

import sys

import mlx.core as mx

from _common import MID_VAL, N, check, survivor_and_trip


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

    if arm == "item":
        v = mid.tolist()[0]
        print(f"outcome={check(v, MID_VAL)} got={v:.6g} want={MID_VAL:.6g}",
              flush=True)
    elif arm == "xstream":
        got = float(mx.sum(mx.add(mid, 1.0)).item())  # ambient default
        want = (MID_VAL + 1.0) * N
        print(f"outcome={check(got, want)} got={got:.6g} want={want:.6g}",
              flush=True)
    elif arm == "samestream":
        got = float(mx.sum(mx.add(mid, 1.0, stream=s), stream=s).item())
        want = (MID_VAL + 1.0) * N
        print(f"outcome={check(got, want)} got={got:.6g} want={want:.6g}",
              flush=True)
    elif arm == "fresh":
        got = float(mx.sum(mx.full((N,), 15.0, stream=s), stream=s).item())
        print(f"outcome={check(got, 15.0 * N)} got={got:.6g}", flush=True)


if __name__ == "__main__":
    main()
