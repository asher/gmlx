"""Reverse topology: producer on the created stream, consumer on the
default stream.

The usual serve shape puts the auto-key producer on the default
stream and the consumer on the worker stream, so any drain that
happens to touch low stream indices first commits the producer first
and stays clean by accident. This probe inverts it: the producer
chain runs on a created (higher-index) stream and the consumer op on
the default stream, so the default stream's buffer carries the fence
wait and the created stream's buffer carries the update.

This decides whether the mlx 0.32.1 unwind handler
(ml-explore/mlx#3675), which synchronizes open streams in stream-set
order (ascending index), commits a fence wait ahead of its update
and reproduces the GPU-timeout abort on the trip itself.

Arms:
  trip-only      - trip and catch, nothing else. Pre-0.32.1
                   (measured): the catch works, then the process
                   aborts AT TEARDOWN with a GPU timeout: exit
                   commits the default stream's open buffer, whose
                   fence wait can never be satisfied. Reverse
                   topology poison is fatal even when nothing else
                   touches the GPU, so this arm is flag-day runbook
                   only pre-fix. Post-fix: exit 0 means the shipped
                   unwind handler survives reverse topology; an
                   abort is the finding that motivates an order-free
                   upstream follow-up.
  producer-first - catch, sync the created stream (update), then the
                   default (wait), then known-answer. Clean on every
                   wheel (measured on 0.31.2): the ordered drain
                   rescues the topology that undrained teardown
                   cannot survive.
  consumer-first - catch, sync the default stream first: commits the
                   wait with its update uncommitted. Pre-0.32.1:
                   abort (flag-day runbook only). Post-fix: clean
                   (the unwind already drained), unless trip-only
                   already aborted.
"""

import sys

import mlx.core as mx

from _common import MID_VAL, N, check, oversized_rows


def main():
    arm = sys.argv[1]

    s = mx.new_stream(mx.default_device())
    x = mx.full((N,), 1.0, stream=s)
    mid = mx.add(mx.multiply(x, 1.0001, stream=s), 1.0, stream=s)
    y = mx.add(mid, 1.0)  # ambient default stream: the cross edge
    big = mx.contiguous(mx.broadcast_to(y, (oversized_rows(), N, N)))
    try:
        mx.eval(big)
        print("outcome=NO-THROW", flush=True)
        return
    except Exception:
        pass

    if arm == "trip-only":
        print("outcome=CAUGHT", flush=True)
        return
    if arm == "producer-first":
        mx.synchronize(s)  # the update's stream first
        mx.synchronize()
    elif arm == "consumer-first":
        mx.synchronize()  # the wait's stream first
        mx.synchronize(s)
    got = float(mx.sum(mx.add(y, 1.0), stream=None).item())
    want = (MID_VAL + 2.0) * N
    print(f"outcome={check(got, want)} got={got:.6g} want={want:.6g}",
          flush=True)


if __name__ == "__main__":
    main()
