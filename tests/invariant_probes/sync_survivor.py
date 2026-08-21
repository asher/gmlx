"""A failed sync eval leaves survivors the scoped drain rehabilitates.

Single-stream tape built under a with-block (deterministic full(),
so no auto-key subgraph and no cross-stream edges). mx.eval throws
at an oversized allocation whose input chain encoded earlier. After
mx.synchronize on the stream, the survivor must read back promptly
and correct: the sync path attaches no events, and the drain commits
and completes the already-encoded kernels.

Expected on every wheel: CLEAN.
"""

import mlx.core as mx

from _common import MID_VAL, N, check, survivor_and_trip


def main():
    s = mx.new_stream(mx.default_device())
    with mx.stream(s):
        mid, big = survivor_and_trip()
    try:
        mx.eval(big)
        print("outcome=NO-THROW", flush=True)
        return
    except Exception:
        pass
    mx.synchronize(s)
    got = float(mx.sum(mid, stream=s).item())
    want = MID_VAL * N
    print(f"outcome={check(got, want)} got={got:.6g} want={want:.6g}",
          flush=True)


if __name__ == "__main__":
    main()
