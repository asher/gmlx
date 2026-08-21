"""Multi-stream tape recovery: peers commit before the thrower.

Ops built with stream= kwargs (not a with-block) put the auto-key
subgraph of mx.random on the ambient default stream, so the tape
spans two streams and the thrower's buffer holds a cross-stream
fence wait whose update sits in the default stream's uncommitted
buffer. Synchronizing the thrower first commits an unsatisfiable
wait: GPU timeout, uncatchable abort. Committing the peer first
makes every wait satisfiable.

Arms:
  orderfix     - bare mx.synchronize() (the peer), then the thrower,
                 then a known-answer graph. Clean on every wheel.
  thrower-first - the misordered drain. Pre-0.32.1: abort (run from
                 the flag-day runbook only). Post-fix: clean, because
                 the 0.32.1 unwind handler already drained both
                 streams before the exception reached the handler.
"""

import sys

import mlx.core as mx

from _common import N, check, oversized_rows


def main():
    arm = sys.argv[1]

    s = mx.new_stream(mx.default_device())
    x = mx.random.normal((N, N), stream=s)  # auto-key on default stream
    mid = mx.add(mx.multiply(x, 1.0001, stream=s), 1.0, stream=s)
    big = mx.contiguous(
        mx.broadcast_to(mid, (oversized_rows(), N, N), stream=s), stream=s
    )
    try:
        mx.eval(big)
        print("outcome=NO-THROW", flush=True)
        return
    except Exception:
        pass
    if arm == "orderfix":
        mx.synchronize()  # the peer (default stream) first
        mx.synchronize(s)
    elif arm == "thrower-first":
        mx.synchronize(s)
        mx.synchronize()
    got = float(mx.sum(mx.full((N,), 15.0, stream=s), stream=s).item())
    print(f"outcome={check(got, 15.0 * N)} got={got:.6g}", flush=True)


if __name__ == "__main__":
    main()
