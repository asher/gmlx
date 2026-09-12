"""Wired-limit sweep control and CPU-stream device setup for streaming
(over-wired-budget) models."""
from __future__ import annotations

import os

import mlx.core as mx


def _neutralize_wired_limit_sweep():
    """Pin the MLX wired limit at its default for the rest of the process.

    mlx-lm wraps generation in a context manager that raises the wired limit
    to the device's max recommended working set. MLX services that by adding
    every live buffer - file-backed zero-copy weight views included - to its
    Metal residency set, which wires them all. For a model larger than the
    wired budget that sweep exhausts wired memory within seconds of the
    first GPU command (hard-panic territory). There is no per-buffer
    opt-out, so streaming mode no-ops ``mx.set_wired_limit`` instead: Metal
    then wires only what GPU work actually references (the every-token
    layers + KV), and
    expert pages stay plain evictable page cache. Covers every caller
    (generate, server batch path, trainer) since all resolve the function
    through ``mx.`` at call time. Idempotent.

    mlx-lm's ``wired_limit()`` context manager also prints a per-generation
    large-model warning sized against the limit this function just pinned -
    meaningless in streaming mode, and noisy (once per chat turn). Swap it
    for a quiet context that keeps the exit synchronize (the original syncs
    the generation stream before restoring the limit; callers may rely on
    that barrier at generator teardown). NB: patched via importlib -
    ``import mlx_lm.generate`` binds the function mlx_lm re-exports in
    ``__init__``, not the submodule.
    """
    if getattr(mx.set_wired_limit, "_kq_no_sweep", False):
        return
    # A generator that started before this call left the limit raised, and
    # its exit restore is a no-op from here on. Lower it now: with it up,
    # the next streaming load's walk wires the file's resident pages as it
    # creates the views, and every command buffer in the process fails
    # with a Metal out-of-memory error.
    try:
        prev = mx.set_wired_limit(0)
    except Exception:
        prev = 0
    if prev:
        print(f"[stream] wired limit lowered from {prev / 1e9:.1f} GB to 0: "
              "a raised limit wires every live buffer, zero-copy views included")

    def _no_sweep(*_a, **_k):
        return 0

    _no_sweep._kq_no_sweep = True
    mx.set_wired_limit = _no_sweep

    import contextlib
    import importlib

    @contextlib.contextmanager
    def _quiet_wired_limit(model, streams=None):
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()

    _quiet_wired_limit._kq_no_sweep = True
    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mod.wired_limit = _quiet_wired_limit


def _install_wired_limit_warn_once():
    """Cap mlx-lm's large-model warning at one print per process.

    ``stream_generate`` enters mlx-lm's ``wired_limit()`` context on every
    call - at least once per chat turn - and on entry the context prints its
    near-the-wired-budget warning unconditionally, so a resident model just
    over the 0.9x threshold re-warns every turn. There is no seam around the
    print, so swap in a re-implementation with identical wiring behavior
    (raise the limit, synchronize on exit, restore) that warns only the
    first time.

    Installed at the end of every ``load_model`` (the resident path); the
    streaming / CPU replacements above are stricter (they drop the sweep
    entirely), so this never overwrites them - and they overwrite this when
    they engage, which is always after load. Idempotent. NB: patched via
    importlib - ``import mlx_lm.generate`` binds the function mlx_lm
    re-exports in ``__init__``, not the submodule.
    """
    import contextlib
    import importlib

    from mlx.utils import tree_reduce

    state = {"warned": False}

    @contextlib.contextmanager
    def _warn_once_wired_limit(model, streams=None):
        if not mx.metal.is_available():
            yield
            return
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            model, 0)
        max_rec_size = mx.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size and not state["warned"]:
            state["warned"] = True
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} "
                f"MB which is close to the maximum recommended size of "
                f"{max_rec_mb} MB. This can be slow. See the documentation "
                "for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)

    _warn_once_wired_limit._kq_warn_once = True
    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "wired_limit", None)
        if fn is None or getattr(fn, "_kq_no_sweep", False) or \
                getattr(fn, "_kq_warn_once", False):
            continue
        mod.wired_limit = _warn_once_wired_limit


def configure_cpu_device():
    """Run everything on the CPU device (``--stream-cpu``): mmap-streamed weights.

    Besides setting the default device this (a) keeps the graph
    single-device - the fused-GDN runtime patch dispatches Metal kernels
    regardless of the default device - and (b) no-ops mlx-lm's
    ``wired_limit`` context: it reads
    ``device_info()["max_recommended_working_set_size"]``, absent on the
    CPU device, and wiring is meaningless on CPU. NB: patched via importlib
    - ``import mlx_lm.generate`` binds the function mlx_lm re-exports in
    ``__init__``, not the submodule.
    """
    import contextlib
    import importlib

    mx.set_default_device(mx.cpu)
    os.environ.setdefault("GMLX_FUSED_GDN", "0")

    @contextlib.contextmanager
    def _wired_noop(model, streams=None):
        yield

    # No sweep at all on CPU; the marker keeps a later load_model's
    # warn-once variant (_install_wired_limit_warn_once) from clobbering it.
    _wired_noop._kq_no_sweep = True
    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mod.wired_limit = _wired_noop
    print("[device] cpu (mmap-streamed weights; fused-GDN Metal patch off)")
