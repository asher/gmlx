"""The residency pool's cold build of a chat model."""

from __future__ import annotations

import contextlib
import os

from .residency_footprint import (_decode_feeder_of, _drafter_bytes,
                                  _plain_footprint, _prefill_ring_of,
                                  _stamp_apc_mode, _stamp_boot_kv_costs,
                                  _streamed_shares, text_adapter_bytes)
from .residency_types import _Entry, _Scratch, _build_scratch, _log


# Archs whose mlx-lm model hand-rolls attention in a way that mishandles the batch
# generator's ragged left-padded mask, so concurrent/ragged batched decode is
# unreliable. The server batches, so warn the operator once per model. gemma2 is the
# lone case: it hand-rolls the +/-50 attn-logit softcap then applies the mask manually
# instead of via scaled_dot_product_attention. Verified upstream, not a gmlx/kq
# bug - single-stream matches llama.cpp token-for-token, and the sdpa-based gemma3 /
# gemma4 batch cleanly (see tests/gen/test_batch_parity.py CANDIDATE_ARCHES note).
_BATCH_UNSAFE_ARCHES = {"gemma2"}


_batch_unsafe_warned: set = set()


def _warn_if_batch_unsafe(model_path: str) -> None:
    """Warn once, to stderr, if ``model_path``'s arch is unreliable under the server's
    ragged batched decode. Best-effort: a header-read failure stays silent (the load
    itself will surface any real problem)."""
    if not model_path or model_path in _batch_unsafe_warned:
        return
    _batch_unsafe_warned.add(model_path)
    try:
        from gguf import GGUFReader

        f = GGUFReader(model_path, "r").fields.get("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f is not None else ""
    except Exception:
        return  # advisory warning only; an unreadable header stays silent
    if arch in _BATCH_UNSAFE_ARCHES:
        _log.warning(
            "%r batched decode is unreliable - mlx-lm hand-rolls %s softcap "
            "attention and its manual mask mishandles the batch generator's "
            "ragged left-padded mask, so concurrent requests can corrupt "
            "output. Single-stream output is correct; serve %s one request "
            "at a time until this is fixed upstream.",
            arch, arch, arch,
        )


class _ChatBuild:
    """The cold build of a chat model, gated before the stock load."""

    def _gate(self, gate_bytes, model_path, *, streaming: bool = False,
              image: bool = False, quiet: bool = False) -> None:
        """U4: a model load and swap take the same headroom check a
        request takes, beside the room the resident streamed models keep.
        Raises with the numbers; GMLX_OVERCOMMIT=1 skips. ``quiet`` raises
        without the log line."""
        from .capacity import preload_gate

        self._hit("measure")
        extra = {"image": True} if image else {}
        room, ceiling, host, floor = self._stream_reservations()
        preload_gate(gate_bytes, str(model_path), streaming=streaming,
                     reserved_bytes=room + ceiling,
                     reserved_ram_bytes=room + host, ram_floor_bytes=floor,
                     quiet=quiet, **extra)

    def _gate_fits(self, gate_bytes, model_path) -> bool:
        """The plain build's gate as a question, with nothing logged."""
        try:
            self._gate(gate_bytes, model_path, quiet=True)
        except RuntimeError:
            return False
        return True

    def _keep_check(self, model_path: str, env=None, build_spec=None):
        """The check a run makes before it loads a MoE
        (``budget.check_wire_with_tables``), for a plain build: its
        resident set beside the mmproj and the MTP drafter that load with
        it, against the memory not wired less the kernel floor, the KV
        room at the serve context and the room the resident streamed
        models keep, with the table-only step first, and again once the
        memory not wired stops rising when it is short
        (``budget.settled_wire_check``). Returns the check, the KV room
        and the bytes it kept beside the room, which the install's chunk
        guard keeps too (``_latch_guarded_wire``). An auto drafter
        (``speculative_auto``) loads only when the experts fit in memory
        beside it, as run's auto drafter does. None for a dense model or
        an unreadable header. A check that fails logs its line."""
        from gmlx.device import stream_budget_bytes
        from gmlx.load.discovery import text_tensor_bytes
        from gmlx.stream import budget as bmod
        from gmlx.stream.table_stream import stream_ple_env

        from .capacity import (deferred_table_bytes, streamed_expert_bytes,
                               streamed_table_bytes)

        if not streamed_expert_bytes(model_path):
            return None
        text = text_tensor_bytes(model_path)
        if not text:
            return None
        deferred = deferred_table_bytes(model_path, env=env)
        tables = (0 if stream_ple_env(env) == "0"
                  else max(0, streamed_table_bytes(model_path) - deferred))
        mmproj = getattr(build_spec, "mmproj", None) or ""
        extra = os.path.getsize(mmproj) if os.path.isfile(mmproj) else 0
        # The adapters wire with the text set.
        extra += text_adapter_bytes(build_spec)
        room = bmod.kv_room_bytes(model_path, env=env, resident=True)
        held, _ceiling, host, _floor = self._stream_reservations()
        drafter = _drafter_bytes(model_path, build_spec)
        auto = bool(getattr(build_spec, "speculative_auto", False))
        check = bmod.settled_wire_check(lambda: bmod.check_wire_with_tables(
            text - deferred + extra, tables, drafter, auto=auto,
            room_bytes=room.bytes + held + host,
            budget=stream_budget_bytes()))
        if not check.fits:
            _log.warning("[stream] %s: %s", model_path, check.line(room))
        return check, room, held + host

    def _tables_for_memory(self, model_path: str, gate_bytes: int,
                           env=None) -> bool:
        """The table-only step of a plain build for the memory free now:
        the gate defers the build, and passes it without the streamable
        tables the budget step leaves in it. The step is then latched for
        the load (``table_stream.stream_tables_for_memory``): the tables
        serve from the file, and the install keeps the experts in memory
        if they fit."""
        from gmlx.stream.table_stream import (stream_ple_env,
                                              stream_tables_for_memory)

        from .capacity import deferred_table_bytes, streamed_table_bytes

        if stream_ple_env(env) == "0":
            return False
        tables = (streamed_table_bytes(model_path)
                  - deferred_table_bytes(model_path, env=env))
        if (tables <= 0 or self._gate_fits(gate_bytes, model_path)
                or not self._gate_fits(gate_bytes - tables, model_path)):
            return False
        stream_tables_for_memory(model_path)
        return True

    def _build_chat(self, cache_key, model_path, adapter_path, model_kind,
                    footprint, *, ttl=None, env=None,
                    build_spec=None) -> _Entry:
        from . import bridge_vlm as _serving

        _warn_if_batch_unsafe(model_path)
        # The capacity table is derived (header-based, no load needed) and
        # gated before the box's biggest allocation starts.
        from .capacity import (
            gathered_table_bytes,
            held_table_bytes,
            install_boot_table,
            predicted_stream_bytes,
            preload_gate_bytes,
        )
        # Before the load, as in the CLI: both stream placements stream
        # the tables and the experts, and the warm touch must not wire
        # them first. Set per build, so a later resident model is touched.
        from gmlx.stream.table_stream import (force_table_stream,
                                              stream_experts_for_memory,
                                              stream_tables_for_memory)

        force_table_stream(
            getattr(build_spec, "stream", None) in ("cpu", "experts"))
        # The text adapters' bytes, added where each figure is used: the
        # repricing below resets gate_bytes from the base.
        adapter_bytes = text_adapter_bytes(build_spec)
        gate_bytes = footprint
        streamed = 0
        shares = (0, 0)
        offloaded = 0
        # Tables whose build fits the budget without them, or the memory
        # free now, on an arch that gathers them per layer: the
        # load maps them and the bridge's install serves them from the
        # file, as a stream build's table share. The experts stay in
        # memory if they fit.
        gathered = 0
        # The leg of the check before the load that streams a plain MoE's
        # experts, as the CLI streams them (``_keep_check``), and the room
        # and bytes it kept beside a set it keeps.
        for_memory = ""
        kept = (None, 0)
        if getattr(build_spec, "stream", None) is None:
            # Each build judges the memory free now afresh.
            _clear_table_step(model_path)
            gate_bytes = _plain_footprint(model_path, footprint, env)
            if isinstance(model_path, str) and model_path.endswith(".gguf"):
                judged = self._keep_check(model_path, env, build_spec)
                keep, kept = ((None, (None, 0)) if judged is None
                              else (judged[0], judged[1:]))
                if keep is not None and keep.mtp_off:
                    # As a run does: the load leaves the auto drafter out,
                    # beside experts kept in memory or streamed.
                    if keep.fits:
                        _log.warning("[mtp] %s: %s", model_path,
                                     keep.mtp_off_line("speculative: true"))
                    env = dict(env or {}, MLX_VLM_GGUF_SPECULATIVE="0")
                if keep is not None and not keep.fits:
                    for_memory = keep.reason
                elif keep is not None and keep.table_bytes:
                    if not gathered_table_bytes(model_path, env=env):
                        stream_tables_for_memory(model_path)
                        gate_bytes = _plain_footprint(model_path, footprint,
                                                      env)
                elif self._tables_for_memory(model_path,
                                             gate_bytes + adapter_bytes, env):
                    gate_bytes = _plain_footprint(model_path, footprint, env)
            if for_memory:
                # Priced and gated as a stream build; the install streams
                # the experts without trying the wire.
                stream_experts_for_memory(model_path, for_memory)
                force_table_stream(True)
                shares = predicted_stream_bytes(str(model_path), env)
                streamed = sum(shares)
                gate_bytes = preload_gate_bytes(footprint, "experts", streamed)
                offloaded = max(shares[0], held_table_bytes(str(model_path)))
            elif isinstance(model_path, str) and model_path.endswith(".gguf"):
                gathered = gathered_table_bytes(model_path, env=env)
                shares = (gathered, 0)
                streamed = gathered
                offloaded = footprint - gate_bytes
        elif getattr(build_spec, "stream", None) == "experts":
            # What the install will stream, by its own ladder: the
            # experts, and the tables with them.
            shares = predicted_stream_bytes(str(model_path), env)
            streamed = sum(shares)
            gate_bytes = preload_gate_bytes(footprint, "experts", streamed)
            offloaded = max(shares[0], held_table_bytes(str(model_path)))
            if streamed:
                _log.info(
                    "preload gate: stream=experts discounts %.1f GB of "
                    "off-disk bytes (streamable tables %.1f GB, routed "
                    "experts %.1f GB; gating on %.1f GB resident)",
                    streamed / 1e9, shares[0] / 1e9, shares[1] / 1e9,
                    gate_bytes / 1e9)
        self._gate(gate_bytes + adapter_bytes, model_path,
                   streaming=(getattr(build_spec, "stream", None) == "experts"
                              or bool(for_memory)))
        guarded = _latch_guarded_wire(
            model_path,
            getattr(build_spec, "stream", None) is None and not for_memory,
            *kept)
        _say_tables_offloaded(model_path, offloaded)
        # What the entry holds resident. A streaming build adds its decode
        # arena below, once the loader has sized it.
        resident_bytes = gate_bytes + adapter_bytes
        # The boot table feeds request admission (width/ctx budgets), so a
        # streaming model's table must also price only the resident share:
        # the expert stacks decode through the disk arena, not the KV budget.
        install_boot_table(str(model_path), resident_bytes, str(model_path),
                           env=env)
        scratch = _Scratch()
        token = _build_scratch.set(scratch)
        # Per-model load-param + APC/SSD-KV env window: set this model's vars
        # around the stock load, then restore. Builds are serialized by
        # ``_build_lock``, so this os.environ mutation never races a sibling load.
        saved_env = self._apply_env(env)
        # Publish the resolved spec for this build so the load bridge can read its
        # profile overrides (e.g. chat_template). ``_stock_get`` blocks here while
        # the engine's generation worker thread runs the bridge, where the
        # request-thread ``_active_spec`` ContextVar is invisible - this module
        # global crosses that boundary. Safe because the build lock serialises
        # builds (one spec live at a time); cleared in the finally.
        _serving.set_build_spec(build_spec)
        # The stock loader's ``apc.from_env`` stays unpatched and unused:
        # capture the effective APC enablement (this model's window value over
        # any ambient shell var - the stock precedence) into GMLX_APC_ENABLED
        # for the bridge's gmlx-owned manager build, then pin APC_ENABLED=0 so
        # from_env returns None instead of building a stock manager (and its
        # disk worker threads) that the bridge's manager would orphan.
        apc_saved = {k: os.environ.get(k)
                     for k in ("GMLX_APC_ENABLED", "APC_ENABLED")}
        enabled = os.environ.get("APC_ENABLED", "0") in (
            "1", "true", "True", "yes")
        os.environ["GMLX_APC_ENABLED"] = "1" if enabled else "0"
        os.environ["APC_ENABLED"] = "0"
        # Untracked-weights ownership window: every registration the stock
        # load performs on this entry's behalf (the model walk, a drafter
        # reload, the MTP stash build) attributes its key to this entry, so
        # _teardown can forget exactly those keys. Same cross-thread module-
        # global pattern as set_build_spec, serialized by the build lock.
        from gmlx.gen.prefill_decay import (
            forget_untracked_weights,
            note_build_credit,
            set_untracked_weights_owner,
        )
        set_untracked_weights_owner(cache_key)
        if streamed > 0:
            # Before the walk, not at the install after it: a resident
            # model's generator left the wired limit raised, and the walk
            # under it wires the file's pages (see the loader).
            from gmlx.stream.wired_limit import _neutralize_wired_limit_sweep
            _neutralize_wired_limit_sweep()
            # The walk wraps the routed experts as tracked MLX views. They
            # are page cache, but the install credits them only at the
            # first request. A governor tick in between reads the file as
            # live memory and sheds another model's rows. Credit what the
            # load adds, up to the streamed bytes, until the install's own
            # credit or deduction takes over. A table no array can hold is
            # read from the file and never adds any.
            note_build_credit(cache_key, streamed - min(
                shares[0], held_table_bytes(str(model_path))))
        try:
            # A raised wired limit would wire the walk's buffers in one
            # burst, ahead of the install's chunk guard.
            from gmlx.stream.wired_limit import wired_limit_held

            with wired_limit_held() if guarded else contextlib.nullcontext():
                self._stock_get(model_path, adapter_path,
                                model_kind=model_kind)
            # One policy resolve per load, inside the env window. An
            # error verdict fails residency. Runs before the APC salt
            # below, which reads the stamp it leaves on the model.
            from .kv_policy import KvPolicyError, resolve_for_load

            rg = scratch.response_generator
            if rg is not None and getattr(rg, "model", None) is not None:
                try:
                    kv_policy = resolve_for_load(rg, model_path)
                except KvPolicyError:
                    raise
                except Exception:
                    kv_policy = None
                    _log.warning("kv policy resolve skipped", exc_info=True)
            else:
                kv_policy = None
            # Wire the bridge-built manager everywhere the stock load would
            # have put a from_env one: the scratch (captured by the _Entry
            # below and served by the runtime proxy), the registry cache dict
            # (read back on model-switch reuse and cleared at unload), and the
            # ResponseGenerator, whose binding is read lazily at the first
            # BatchGenerator build - this runs before any request reaches it.
            manager = _serving.pop_built_apc_manager()
            if manager is not None:
                # The kvarn wire salt could not be set at manager build
                # (pre-load, nothing to probe); set it here from the
                # stamped policy, gated on the loaded model actually
                # converting. Every manager passes through this block:
                # GMLX_APC_ENABLED is set only above, and
                # build_apc_manager returns None without it.
                from gmlx.cache.kvarn_apc import apply_kvarn_salt
                rg = scratch.response_generator
                model = getattr(rg, "model", None)
                if model is None:
                    mc = scratch.model_cache
                    model = mc.get("model") if hasattr(mc, "get") else None
                apply_kvarn_salt(manager, model)
                scratch.apc_manager = manager
                mc = scratch.model_cache
                if hasattr(mc, "ensure_kind"):
                    mc.ensure_kind()["apc_manager"] = manager
                elif isinstance(mc, dict):
                    mc["apc_manager"] = manager
                if rg is not None:
                    rg.apc_manager = manager
                    _stamp_apc_mode(rg)
                try:
                    manager.autosize(getattr(rg, "model", None))
                except Exception:
                    _log.warning("APC pool autosize skipped", exc_info=True)
            _stamp_boot_kv_costs(rg, str(model_path))
            # The install decided on the loaded model; price what it
            # really streamed if the header prediction differed.
            repriced = priced_up = False
            if (getattr(build_spec, "stream", None) == "experts" or gathered
                    or for_memory or guarded):
                done = _streamed_shares(getattr(rg, "model", None), shares,
                                        str(model_path))
                if done is None and for_memory:
                    # No install read the latch: the experts are in memory.
                    stream_experts_for_memory(model_path)
                    _log.warning(
                        "[capacity] %s holds its experts in memory, since "
                        "this load has no streaming install; re-pricing "
                        "the resident weights", model_path)
                    gate_bytes = _plain_footprint(model_path, footprint, env)
                    resident_bytes = gate_bytes + adapter_bytes
                    repriced = priced_up = True
                elif done is not None and sum(done) != streamed:
                    _log.warning(
                        "[capacity] %s streams %.1f GB (tables %.1f GB, "
                        "experts %.1f GB), not the %.1f GB the header "
                        "predicted; re-pricing the resident weights",
                        model_path, sum(done) / 1e9, done[0] / 1e9,
                        done[1] / 1e9, streamed / 1e9)
                    gate_bytes = preload_gate_bytes(
                        footprint, "experts", sum(done))
                    resident_bytes = gate_bytes + adapter_bytes
                    repriced = True
                    priced_up = sum(done) < streamed
            # The decode arena is MLX-tracked and sized at install, after
            # the pre-load table priced the resident weights alone. Price
            # it in now so admission, /v1/estimate and the boot refusal
            # see the KV room that is really left.
            feeder = _decode_feeder_of(rg)
            if feeder is not None and getattr(feeder, "nominal_bytes", 0):
                resident_bytes = (gate_bytes + adapter_bytes
                                  + int(feeder.nominal_bytes)
                                  + _prefill_ring_of(rg))
                try:
                    install_boot_table(
                        str(model_path), resident_bytes,
                        str(model_path), env=env)
                except RuntimeError as e:
                    _log.warning("[capacity] the decode arena leaves no "
                                 "context at width 1 (GMLX_DECODE_ARENA_GB "
                                 "lowers it): %s", e)
            elif repriced:
                install_boot_table(str(model_path), resident_bytes,
                                   str(model_path), env=env)
        except BaseException as e:
            # A failed build never reaches _teardown: drop its partial
            # registrations here or they tax headroom forever.
            forget_untracked_weights(cache_key)
            _log.warning("model build failed: %s", model_path, exc_info=True)
            self._discard_failed_build(cache_key, model_path, scratch)
            rg = feeder = None
            # The frames below hold the half-built model and its views.
            # The next build's gate reads them as live memory until they
            # go; detach them so acquire's collect frees them first.
            e.__traceback__ = None
            raise e
        finally:
            set_untracked_weights_owner(None)
            _serving.set_build_spec(None)
            _serving.pop_built_apc_manager()
            self._restore_env(apc_saved)
            self._restore_env(saved_env)
            _build_scratch.reset(token)
        return self._new_entry(
            cache_key, model_path, resident_bytes, ttl,
            model_cache=scratch.model_cache,
            response_generator=scratch.response_generator,
            apc_manager=scratch.apc_manager,
            kv_policy=kv_policy,
            priced_up=priced_up,
        )

    @staticmethod
    def _apply_env(env):
        """Set ``env`` in ``os.environ``, returning the prior values for restore."""
        if not env:
            return None
        saved = {}
        for k, v in env.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return saved

    @staticmethod
    def _restore_env(saved):
        if not saved:
            return
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _say_tables_offloaded(model_path, table_bytes: int) -> None:
    """The server log's [stream] line of a build that reads
    ``table_bytes`` of its tables from the file. Nothing for 0."""
    if table_bytes > 0:
        from gmlx.stream.budget import tables_offloaded

        from .capacity import table_label

        _log.warning("[stream] %s: %s", model_path,
                     tables_offloaded(table_label(str(model_path)),
                                      table_bytes))


def _clear_table_step(model_path) -> None:
    """Drop the table-only step a plain build latched for ``model_path``
    (``_ResidencyPool._tables_for_memory``). It holds while the model it
    judged is resident, so admission prices a later build of the path
    from its file."""
    if isinstance(model_path, str) and model_path.endswith(".gguf"):
        from gmlx.stream.table_stream import (stream_experts_for_memory,
                                              stream_tables_for_memory)

        stream_tables_for_memory(model_path, False)
        stream_experts_for_memory(model_path)


def _latch_guarded_wire(model_path, plain: bool, room=None,
                        extra: int = 0) -> bool:
    """Latch the chunked wire for a plain build of a MoE the gate admitted
    (``table_stream.guard_resident_wire``): its install wires it in
    chunks under the chunk guard, keeping the ``room`` and ``extra``
    bytes the check before the load kept, and the warm touch leaves it
    alone. Cleared for any other build of the path, and for a diffusion
    model, which the bridge hands back with no install. True when
    latched."""
    if isinstance(model_path, str) and model_path.endswith(".gguf"):
        from gmlx.gen.diffusion import DIFFUSION_ARCHES
        from gmlx.serve.capacity import _table_shards, streamed_expert_bytes
        from gmlx.stream.table_stream import guard_resident_wire

        on = (plain and bool(streamed_expert_bytes(model_path))
              and _table_shards(model_path)[1] not in DIFFUSION_ARCHES)
        guard_resident_wire(model_path, on, room, extra)
        return on
    return False
