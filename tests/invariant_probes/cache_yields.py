"""The buffer cache yields at the memory limit.

On a cache miss the Metal allocator releases cached buffers until
active plus cache plus the new buffer fit under its reclaim threshold,
the lower of the memory limit and 0.95 of the working set. Under a
memory limit, active plus cache therefore stays under it across
allocations and frees, as long as active alone does. An allocation
that active alone pushes past the limit is made anyway, with the cache
emptied first: the limit never throws.

Arm ``limited`` sets the memory limit 64 MiB above the bytes held at
start; arm ``default`` keeps MLX's own limit, to show the same steps
then hold more than 64 MiB. Prints the peak of active plus cache read
after each step, and for ``limited`` the outcome of an allocation past
the limit.
"""

import sys

import mlx.core as mx

MIB = 1 << 20


def main(arm: str) -> None:
    mx.set_cache_limit(1 << 40)  # the pool's own limit never trims it
    mx.synchronize()
    base = mx.get_active_memory() + mx.get_cache_memory()
    limit = base + 64 * MIB
    if arm == "limited":
        mx.set_memory_limit(limit)
    peak = [0]

    def settle():
        mx.synchronize()
        peak[0] = max(peak[0], mx.get_active_memory() + mx.get_cache_memory())

    def alloc(mib):
        # Sizes far enough apart that the cache never reuses one for
        # another, so each is a cache miss.
        a = mx.zeros((mib * MIB,), dtype=mx.uint8)
        mx.eval(a)
        settle()
        return a

    a = alloc(24)
    b = alloc(24)
    del a
    settle()  # 24 MiB cached
    c = alloc(30)  # a miss that needs 78 MiB
    del b, c
    settle()  # 54 MiB cached
    d = alloc(40)  # a miss that needs 94 MiB
    del d
    settle()
    held = peak[0]
    over = "-"
    if arm == "limited":
        try:
            # Past the limit with the cache emptied first; the fill's
            # scalar may leave a few bytes cached.
            e = alloc(96)
            over = ("made" if mx.get_active_memory() > limit
                    and mx.get_cache_memory() < MIB else "short")
            del e
        except Exception as exc:  # the limit must never throw
            over = f"THROW:{type(exc).__name__}"
    print(f"arm={arm} limit={limit} peak={held} over={over}")


if __name__ == "__main__":
    main(sys.argv[1])
