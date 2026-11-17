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
# The one hold that every wired_limit_held block in force shares: how many
# blocks hold it, the limit asked for last while they do, and the function
# the last block out puts back. _HOLD_LOCK orders every edit of
# mx.set_wired_limit (a hold's start and end, the cap's install, the pin)
# and the recorder's read of the count. No writer takes LIMIT_LOCK under
# it, and no holder of LIMIT_LOCK sets the wired limit.
_HOLD_LOCK = threading.RLock()
_HOLDS = 0
_WANT = 0
_REAL = None


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


def _recorder(limit, *_a, **_k):
    """``mx.set_wired_limit`` while a hold is in force: record the ask and
    return the limit asked for before it. A reference taken during a hold
    and called after the last one ends goes to the function in place."""
    global _WANT
    with _HOLD_LOCK:
        if _HOLDS:
            prev, _WANT = _WANT, int(limit)
            return prev
        fn = mx.set_wired_limit
    return fn(limit, *_a, **_k)


def _hold_start() -> bool:
    """Count one more hold. The first one in reads the limit, sets 0 and
    installs the recorder. False when no hold starts: the sweep is
    neutralized, or the limit cannot be read."""
    global _HOLDS, _WANT, _REAL
    with _HOLD_LOCK:
        if wrapped_with(mx.set_wired_limit, "_kq_no_sweep"):
            return False
        if _HOLDS:
            _HOLDS += 1
            return True
        real = mx.set_wired_limit
        try:
            want = int(real(0) or 0)
        except Exception:
            return False
        _REAL, _WANT, _HOLDS = real, want, 1
        mx.set_wired_limit = link(_recorder, real)
        return True


def _hold_end() -> None:
    """Count one hold less. The last one out puts back the function under
    the recorder and applies the limit asked for last. After a pin inside
    the hold (``_neutralize_wired_limit_sweep``) the pin stands."""
    global _HOLDS
    with _HOLD_LOCK:
        _HOLDS -= 1
        if _HOLDS or mx.set_wired_limit is not _recorder:
            return
        mx.set_wired_limit = _REAL
        if _WANT:
            _REAL(_WANT)


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
    several threads share one hold, which starts and ends under one lock:
    the first block in reads the limit and sets 0, and the last one out,
    whichever started it, puts back the function and the limit last asked
    for."""
    if not _hold_start():
        yield
        return
    try:
        yield
    finally:
        _hold_end()


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
    global _ANNOUNCE, _REAL
    if announce:
        _ANNOUNCE = True
    from gmlx.serve import capacity

    with _HOLD_LOCK:
        if (wrapped_with(mx.set_wired_limit, "_kq_wired_cap")
                or wrapped_with(mx.set_wired_limit, "_kq_no_sweep")):
            return
        cap = capacity.wired_cap_bytes()
        if cap is None:
            return
        # Inside a hold, the clamp goes under it, on what the hold puts
        # back.
        held = mx.set_wired_limit is _recorder
        capped = _clamp(_REAL if held else mx.set_wired_limit, capacity)
        if held:
            _REAL = capped
            link(_recorder, capped)
        else:
            mx.set_wired_limit = capped
    _limit_memory_to_cap(cap)


def _clamp(real, capacity):
    """``real`` clamped to the live wired cap."""

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

    return link(_capped, real, "_kq_wired_cap")


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
    def _no_sweep(*_a, **_k):
        return 0

    _no_sweep._kq_no_sweep = True
    with _HOLD_LOCK:
        if wrapped_with(mx.set_wired_limit, "_kq_no_sweep"):
            return
        # A generator that started before this call left the limit raised,
        # and its exit restore is a no-op from here on. Lower it now: with
        # it up, the next streaming load's walk wires the file's resident
        # pages as it creates the views, and every command buffer in the
        # process fails with a Metal out-of-memory error.
        try:
            prev = mx.set_wired_limit(0)
        except Exception:
            prev = 0
        mx.set_wired_limit = _no_sweep
    if prev:
        print(f"[stream] wired limit lowered from {prev / 1e9:.1f} GB to 0: "
              "a raised limit wires every live buffer, zero-copy views included")
    # MLX wires nothing from here on, and the file-backed views a pinned
    # load tracks count as active bytes: under a memory limit at the wired
    # cap each eval would wait for the GPU work in flight.
    _restore_mlx_memory_limit()

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
