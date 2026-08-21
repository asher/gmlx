"""Thread/stream ownership rules.

Metal command encoders are thread_local, so a created stream is
usable only from the thread that created it. Cross-thread use throws
("There is no Stream(gpu, N) in current thread"); it does not hang
or corrupt. The default stream is usable from any thread.

Cases (each spawned case joined with a timeout; HANG if it never
returns):
  a: eval on the default stream from a spawned thread   -> ok
  b: new_stream created inside the thread, eval on it   -> ok
  c: eval (from thread) of a graph on a main stream     -> THROW
  d: mx.synchronize(main-created stream) from a thread  -> THROW
  e: main evals a graph on a thread-created stream      -> THROW

Same outcomes expected on every wheel.
"""

import threading

import mlx.core as mx


def run(name, fn):
    out = {}

    def work():
        try:
            fn()
            out["r"] = "ok"
        except Exception as e:
            out["r"] = f"THROW {type(e).__name__}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        out["r"] = "HANG"
    print(f"case={name} outcome={out.get('r')}", flush=True)


def case_a():
    mx.eval(mx.ones((256, 256)) @ mx.ones((256, 256)))


def case_b():
    s = mx.new_stream(mx.default_device())
    k = mx.full((256, 256), 3.0, stream=s)
    mx.eval(mx.sum(k, stream=s))


def main():
    s_main = mx.new_stream(mx.default_device())
    g_main = mx.sum(mx.full((256, 256), 2.0, stream=s_main), stream=s_main)

    run("a", case_a)
    run("b", case_b)
    run("c", lambda: mx.eval(g_main))
    run("d", lambda: mx.synchronize(s_main))

    holder = {}

    def make():
        holder["s"] = mx.new_stream(mx.default_device())
        holder["g"] = mx.sum(
            mx.full((256, 256), 4.0, stream=holder["s"]), stream=holder["s"]
        )

    t = threading.Thread(target=make, daemon=True)
    t.start()
    t.join()
    try:
        mx.eval(holder["g"])
        print("case=e outcome=ok", flush=True)
    except Exception as e:
        print(f"case=e outcome=THROW {type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()
