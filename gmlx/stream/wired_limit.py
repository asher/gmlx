"""Wired-limit sweep control and CPU-stream device setup for streaming
(over-wired-budget) models."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading

import mlx.core as mx

from gmlx.upstream.patching import link, wrapped_with

_log = logging.getLogger(__name__)

# (MLX's memory limit as install_wired_cap found it, the wired cap it set
# in its place); the pin puts the first back while the second stands.
# None until a cap sets the limit.
_CAP_MEMORY_LIMIT: tuple[int, int] | None = None
# True once the clamp has told of a request it lowered.
_CLAMP_LOGGED = False
# True when the cap's lines print (the server); else they show under
# --verbose only.
_ANNOUNCE = False
# Held by each writer that reads the MLX memory limit and then sets it
# (the cap, its restore at a pin, the governor's throttle, expert
# streaming's allocator limits), so none sets a limit over one another
# thread set between its read and its write. The throttle and expert
# streaming set the cache limit under it too. The image cache bound
# (image/memfit.py) and the image command swap the cache limit outside
# it: each puts back the limit it saw on entry, and a lock would not
# order those swaps against a throttle that arms and disarms between.
LIMIT_LOCK = threading.RLock()


def _say(msg: str) -> None:
    if _ANNOUNCE:
        print(msg)
    else:
        from gmlx.load import loadlog

        loadlog.info(msg)


@contextlib.contextmanager
def cpu_wired_limit(streams=None):
    """The wired limit context on the CPU device: no limit, and the exit
    synchronize of the copy's context."""
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()


@contextlib.contextmanager
def wired_limit_held():
    """Hold the MLX wired limit at 0 for the block, then apply the limit
    the process last asked for. Under a raised limit MLX wires every
    buffer a load creates, so a load beside another model's generation, or
    an MTP model's sticky wire, would wire its weights in one burst, before
    the chunked wire's guard reads memory. Inside the block, calls to
    ``mx.set_wired_limit`` (a generation starting or ending) are recorded
    and return the limit last asked for. A sweep neutralized inside the
    block (``_neutralize_wired_limit_sweep``, a load that streams) keeps
    the limit at 0. A no-op once the sweep is neutralized. Blocks on
    other threads may end in either order: a block that ends under a later
    one still in force leaves the chain, and the later one's end puts back
    the function under both and the limit last asked for."""
    if wrapped_with(mx.set_wired_limit, "_kq_no_sweep"):
        yield
        return
    # The function the block puts back, in a cell: the wired cap installed
    # inside the block goes under the hold (install_wired_cap).
    real = [mx.set_wired_limit]
    try:
        want = [int(real[0](0) or 0)]
    except Exception:
        yield
        return

    def _held(limit, *_a, **_k):
        prev, want[0] = want[0], int(limit)
        return prev

    _held._kq_wired_real = real
    mx.set_wired_limit = link(_held, real[0])
    try:
        yield
    finally:
        if mx.set_wired_limit is _held:
            mx.set_wired_limit = real[0]
            if want[0]:
                real[0](want[0])
        else:
            # A later block is still in force: the block above this one
            # takes the function under it. Its start read this block's
            # limit, and every ask since went to it, so it ends on the
            # limit last asked for.
            above = mx.set_wired_limit
            while hasattr(above, "_kq_wired_real"):
                cell = above._kq_wired_real
                if cell[0] is _held:
                    cell[0] = real[0]
                    link(above, real[0])
                    break
                above = cell[0]


def install_wired_cap(announce: bool = False) -> None:
    """Limit this process under the wired cap for its life, once
    ``capacity.decide_cap`` puts it there, before its first load: clamp
    every ``mx.set_wired_limit`` call to the cap
    (``capacity.wired_cap_bytes``) and set the MLX memory limit to it,
    once per process. gmlx sets the wired limit in many places
    (the generation contexts of the copied mlx-lm and mlx-vlm, the MTP
    sticky wire, the trainers, the CLI verbs), all through ``mx.`` at
    call time, so this one wrapper owns the figure each gets:
    min(request, cap). Under the memory limit, MLX frees cached buffers
    before its active bytes and its buffer cache together pass the cap,
    so the cache takes no fixed share of it. The memory limit keeps the
    cap's startup figure while the clamp follows the live cap, which
    memory wired outside the process can hold lower. A pinned process
    (``_neutralize_wired_limit_sweep``) stays at 0 and keeps MLX's memory
    limit, and ``wired_limit_held`` holds and restores through the clamp.
    ``announce`` prints the cap and the first request the clamp lowers;
    else they show under --verbose. A process not under the cap never
    calls this, and keeps MLX's limits. A no-op without Metal."""
    global _ANNOUNCE
    if announce:
        _ANNOUNCE = True
    from gmlx.serve import capacity

    if (wrapped_with(mx.set_wired_limit, "_kq_wired_cap")
            or wrapped_with(mx.set_wired_limit, "_kq_no_sweep")):
        return
    cap = capacity.wired_cap_bytes()
    if cap is None:
        return
    # Inside a hold, the clamp goes under it, on what the hold puts back.
    held, real = None, mx.set_wired_limit
    while hasattr(real, "_kq_wired_real"):
        held, real = real, real._kq_wired_real[0]

    def _capped(limit, *a, **k):
        global _CLAMP_LOGGED
        cap = capacity.wired_cap_bytes()
        if cap is not None and int(limit) > int(cap):
            if not _CLAMP_LOGGED:
                _CLAMP_LOGGED = True
                _say(f"[memory] wired limit {int(limit) / 1e9:.1f} GB "
                     f"asked, set to the wired cap {cap / 1e9:.1f} GB")
            limit = int(cap)
        return real(limit, *a, **k)

    capped = link(_capped, real, "_kq_wired_cap")
    if held is None:
        mx.set_wired_limit = capped
    else:
        held._kq_wired_real[0] = capped
        link(held, capped)
    _limit_memory_to_cap(cap)


def _limit_memory_to_cap(cap: float) -> None:
    """Set the MLX memory limit to the wired cap, and log both. A cap of
    0 (more memory wired outside the server than the RAM term leaves)
    keeps MLX's limit: under a limit of 0 every allocation would empty the
    buffer cache and every eval would wait for the GPU."""
    global _CAP_MEMORY_LIMIT
    from gmlx.serve import capacity

    ws = capacity.working_set_bytes() or 0.0
    why = (f"working set {ws / 1e9:.1f} GB less {capacity.margin():.0%}, or "
           f"RAM less the reserve and the "
           f"{capacity.outside_wired_bytes() / 1e9:.1f} GB wired outside "
           f"the server, whichever is lower")
    if cap <= 0:
        print(f"[memory] warning: wired cap 0 GB ({why}): the MLX memory "
              "limit stays at its default", file=sys.stderr)
        return
    try:
        with LIMIT_LOCK:
            prev = int(mx.set_memory_limit(int(cap)))
            _CAP_MEMORY_LIMIT = (prev, int(cap))
    except Exception:
        _log.warning("[memory] set_memory_limit failed; the MLX memory "
                     "limit stays at its default", exc_info=True)
        return
    _say(f"[memory] wired cap {cap / 1e9:.1f} GB ({why}): "
         f"mx.set_wired_limit clamped to it, MLX memory limit "
         f"{int(cap) / 1e9:.1f} GB (was {prev / 1e9:.1f} GB)")


def note_untracked_tables(names, nbytes: float) -> None:
    """Print, once a load's install is done, the limits its tables mapped
    outside MLX's tracker leave, read back: who owns ``mx.set_wired_limit``
    (the clamp at the wired cap, or a pin at 0) and the MLX memory limit.
    With the clamp and the memory limit at the same figure, the load keeps
    its weights and KV wired under the cap."""
    from gmlx.load import loadlog
    from gmlx.serve import capacity

    try:
        limit = f"{int(mx.get_memory_limit()) / 1e9:.1f} GB"
    except Exception:
        limit = "unread"
    if wrapped_with(mx.set_wired_limit, "_kq_no_sweep"):
        wired = "the wired limit is pinned at 0"
    elif wrapped_with(mx.set_wired_limit, "_kq_wired_cap"):
        cap = capacity.wired_cap_bytes() or 0.0
        wired = ("mx.set_wired_limit clamped to the wired cap "
                 f"{cap / 1e9:.1f} GB")
    else:
        wired = "mx.set_wired_limit unclamped"
    loadlog.note(f"[memory] {', '.join(names)} untracked, "
                 f"{nbytes / 1e9:.1f} GB: {wired}, MLX memory limit {limit}")


def _restore_mlx_memory_limit() -> None:
    """Put back the MLX memory limit install_wired_cap replaced, while
    the cap it set still stands (a later owner, such as expert
    streaming's allocator, keeps its own). While another limit stands,
    the record stays: the governor's throttle, disarmed, puts the cap
    back over its own limit and calls this again."""
    global _CAP_MEMORY_LIMIT
    with LIMIT_LOCK:
        saved = _CAP_MEMORY_LIMIT
        if saved is None:
            return
        try:
            if int(mx.get_memory_limit()) != saved[1]:
                return
            mx.set_memory_limit(saved[0])
        except Exception:
            return
        _CAP_MEMORY_LIMIT = None


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
    if wrapped_with(mx.set_wired_limit, "_kq_no_sweep"):
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
    # MLX wires nothing from here on, and the file-backed views a pinned
    # load tracks count as active bytes: under a memory limit at the wired
    # cap each eval would wait for the GPU work in flight.
    _restore_mlx_memory_limit()

    def _no_sweep(*_a, **_k):
        return 0

    _no_sweep._kq_no_sweep = True
    mx.set_wired_limit = _no_sweep

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
    for mod_name in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils"):
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
    import importlib

    from mlx.utils import tree_reduce

    from gmlx.device import cpu_default_device, metal_value

    state = {"warned": False}

    @contextlib.contextmanager
    def _warn_once_wired_limit(model, streams=None):
        if not mx.metal.is_available():
            yield
            return
        # The CPU device's info has no working set key, and a generation
        # there has nothing to wire, as in the copy's context.
        if cpu_default_device():
            with cpu_wired_limit(streams):
                yield
            return
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            model, 0)
        max_rec_size = metal_value("max_recommended_working_set_size")
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
    for mod_name in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "wired_limit", None)
        if fn is None or wrapped_with(fn, "_kq_no_sweep") or \
                wrapped_with(fn, "_kq_warn_once"):
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
    import importlib

    mx.set_default_device(mx.cpu)
    os.environ.setdefault("GMLX_FUSED_GDN", "0")

    @contextlib.contextmanager
    def _wired_noop(model, streams=None):
        yield

    # No sweep at all on CPU; the marker keeps a later load_model's
    # warn-once variant (_install_wired_limit_warn_once) from clobbering it.
    _wired_noop._kq_no_sweep = True
    for mod_name in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mod.wired_limit = _wired_noop
    print("[device] cpu (mmap-streamed weights; fused-GDN Metal patch off)")
