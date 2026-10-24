#!/usr/bin/env python3
"""The resident wire check (gmlx.stream.budget.check_resident_wire) and
the table-only install's chunked expert wire
(expert_streaming._wire_resident_experts), which commits each chunk once
reclaimable memory covers the check's keep and the chunk. Every memory
counter is scripted: the box from host_box, the vm from the case."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import host_box
import gmlx.stream.budget as budget
import gmlx.stream.expert_streaming as es
import gmlx.stream.table_stream as ts
from gmlx.stream.expert_streaming import install_expert_streaming

GiB = 1 << 30


def _box(box, *, rec, fast):
    """``box`` with reclaimable RAM ``rec`` and free, purgeable and
    speculative RAM ``fast`` (the rest of rec is file-backed)."""
    mem = box["memory_size"]
    host_box.use(box, {
        "free": int(fast), "inactive": 0, "purgeable": 0, "speculative": 0,
        "filebacked": int(rec - fast), "wired": mem // 20,
        "compressor": 0})


class _Clock:
    """The guard's clock: a sleep moves it on at once. Every other
    attribute is the time module's."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def __getattr__(self, name):
        import time

        return getattr(time, name)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GMLX_OVERCOMMIT", "GMLX_WIRE_CHUNK_GB",
              "GMLX_GOV_KERNEL_FLOOR_GB", "GMLX_DECODE_RAM_FLOOR_GB",
              "GMLX_DECODE_PAGECACHE_GB", "GMLX_DECODE_PROCESS_GB",
              "GMLX_DECODE_RESIDENT_CACHE_GB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(es, "time", _Clock())


def _check(w, room=0, **kw):
    return budget.check_resident_wire(int(w), room_bytes=int(room), **kw)


# --- the check on small and large boxes -----------------------------------


def test_the_ci_runner_wires_a_tiny_gguf():
    # The runner's counters leave less than the capped keep: a wire of
    # one chunk or less is judged on the budget only.
    box, vm = host_box.runner()
    host_box.use(box, vm)
    assert budget.live_reclaimable_ram_bytes() < budget.ram_bytes() // 5
    assert _check(10_000_000, room=64 << 20).fits


@pytest.mark.parametrize("chunk_gb", ["", "200"])
def test_a_wire_one_byte_over_2_gib_takes_the_reclaimable_leg(monkeypatch,
                                                             chunk_gb):
    # GMLX_WIRE_CHUNK_GB sets the chunk size only.
    monkeypatch.setenv("GMLX_WIRE_CHUNK_GB", chunk_gb)
    box, _ = host_box.runner()
    _box(box, rec=0.5 * GiB, fast=0.5 * GiB)
    assert _check(2 * GiB).fits
    check = _check(2 * GiB + 1)
    assert check.reason == "reclaimable" and check.reclaimable_bytes


def test_a_big_wire_chunk_still_takes_both_live_legs(monkeypatch):
    # GMLX_WIRE_CHUNK_GB=200 on a 256 GiB Mac: a 150 GB set is judged on
    # reclaimable memory before its load, and its wire, one 200 GiB
    # chunk, is still guarded.
    monkeypatch.setenv("GMLX_WIRE_CHUNK_GB", "200")
    box, _ = host_box.mac(256)
    _box(box, rec=120 * GiB, fast=100 * GiB)
    assert _check(150e9).reason == "reclaimable"
    _box(box, rec=200 * GiB, fast=150 * GiB)
    assert _check(150e9).fits
    seen = {}

    def residency(model, arrays, **kw):
        seen.update(kw)
        return True

    monkeypatch.setattr(es, "_install_gpu_residency", residency)
    monkeypatch.setattr(budget, "kv_room_bytes",
                        lambda p, **kw: budget.KvRoom(0, 32768, 1, 0, 0, 0,
                                                      priced=True))
    assert es._wire_resident_experts(object(), "m.gguf", int(150e9),
                                     budget=None, prefill_step=None,
                                     skip_ids=frozenset())
    assert seen["chunk_bytes"] == 200 * GiB
    assert isinstance(seen["guard"], es._ChunkGuard)


def test_a_16gb_box_keeps_its_moe_resident():
    # A 7 GB MoE at a small run room on a mostly idle 16 GB Mac. The arena's
    # 9.5 GiB of fixed floor and overhead would stream it.
    box, _ = host_box.mac(16)
    _box(box, rec=12 * GiB, fast=10 * GiB)
    check = _check(7e9, room=1 * GiB)
    assert check.fits
    assert check.keep_bytes == int(budget.kernel_floor_bytes()) + GiB


def test_the_keep_is_the_kernel_floor_and_the_room():
    # The check reads reclaimable memory after the load: the arena's host
    # floor and host overhead are out of it already.
    box, _ = host_box.mac(128)
    _box(box, rec=100 * GiB, fast=100 * GiB)
    assert budget.kernel_floor_bytes() == 4e9
    assert _check(50 * GiB, room=8 * GiB).keep_bytes == 4e9 + 8 * GiB


def test_a_128gb_box_with_22gib_of_apps_keeps_flash_next_resident():
    # Flash-Next's resident set (every-token weights and experts, 82.5 GB)
    # beside 22 GiB of other processes' anonymous memory: 101.7 GB
    # reclaimable, 89.0 GB once the floor and the run room are kept.
    box, _ = host_box.mac(128)
    _box(box, rec=94.7 * GiB, fast=76.2 * GiB)
    assert _check(82.5e9, room=8.1 * GiB, budget=int(108.2e9)).fits


@pytest.mark.parametrize("chunked", [False, True])
def test_flash_next_streams_when_apps_hold_32gib(chunked):
    # The same set beside 32 GiB of other processes' memory.
    # GMLX_OVERCOMMIT=1 still stops a wire in chunks, and holds the
    # experts of any other load.
    box, _ = host_box.mac(128)
    _box(box, rec=85 * GiB, fast=66.5 * GiB)
    check = _check(82.5e9, room=8.1 * GiB, budget=int(108.2e9))
    assert check.reason == "reclaimable"
    room = budget.KvRoom(int(8.1 * GiB), 4096, 1, 0, 0, 0, priced=True)
    lever = "GMLX_OVERCOMMIT=1 keeps them in memory" + (
        " unless the memory the kernel can reclaim runs out while they wire."
        if chunked else ".")
    assert check.line(room, 4096, chunked=chunked) == (
        f"{budget.STREAM_FREE_NOW}: holding them takes 82.5 GB, against "
        "91.3 GB of reclaimable memory less 12.7 GB kept for the kernel "
        "floor and the KV room of 8.7 GB at 4096 tokens (the run's prompt "
        f"and max tokens). To keep them in memory, close other apps. {lever}")


def test_an_idle_128gb_box_keeps_the_experts_resident_at_the_run_room():
    box, _ = host_box.mac(128)
    _box(box, rec=110 * GiB, fast=100 * GiB)
    assert _check(82.5e9, room=8.1 * GiB, budget=int(108.2e9)).fits


def test_a_full_page_cache_counts_as_reclaimable():
    # A 128 GB Mac with another model's pages cached: 42.3 GiB free,
    # purgeable and speculative, 54.6 GiB of file pages.
    box, _ = host_box.mac(128)
    _box(box, rec=96.9 * GiB, fast=42.3 * GiB)
    assert _check(70 * GiB, room=8.1 * GiB, budget=int(108.2e9)).fits


def test_the_budget_leg_judges_every_wire():
    box, vm = host_box.runner()
    host_box.use(box, vm)
    assert _check(1000, budget=999).reason == "budget"


def test_overcommit_admits(monkeypatch):
    box, _ = host_box.mac(128)
    _box(box, rec=10 * GiB, fast=1 * GiB)
    assert not _check(82.5e9).fits
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert _check(82.5e9).fits


def test_the_model_s_cached_pages_earn_no_credit():
    # A wired page leaves the file-backed count whether or not it was
    # cached, so an 87 GiB model fully cached in 80 GiB of file pages
    # still takes 87 out of the reclaimable sum.
    box, _ = host_box.mac(128)
    _box(box, rec=90 * GiB, fast=10 * GiB)
    assert _check(87 * GiB).reason == "reclaimable"


def test_the_room_is_kept():
    box, _ = host_box.mac(128)
    _box(box, rec=100 * GiB, fast=100 * GiB)
    w = 92 * GiB              # fits beside the kernel floor
    assert _check(w).fits
    assert _check(w, room=8 * GiB).reason == "reclaimable"


def test_the_troubleshooting_page_is_filed_under_the_lines_a_load_prints():
    # A search for the line a load printed finds its entry.
    page = Path(__file__).parents[2] / "docs" / "troubleshooting.md"
    heads = [h for h in page.read_text().splitlines() if h.startswith("### ")]
    mtp = budget.WireCheck(1, None, 0, None, dropped_bytes=GiB).mtp_off_line()
    for sentence in (budget.STREAM_FREE_NOW, budget.STREAM_OVER_BUDGET,
                     mtp.split(", and the ")[0]):
        assert any(f"`{sentence}`" in h for h in heads), sentence


# --- the chunked wire -----------------------------------------------------


class _Residency:
    """mlx_kquant's residency set: changes stage until a commit applies
    them in order. ``on_commit`` sees the bytes each commit added."""

    def __init__(self, monkeypatch, on_commit=None):
        import mlx_kquant as kq

        self.members: dict = {}
        self.staged: list = []
        self.commits: list = []
        self.on_commit = on_commit
        for name, fn in (("residency_insert", self.insert),
                         ("residency_erase", self.erase),
                         ("residency_commit", self.commit)):
            monkeypatch.setattr(kq, name, fn, raising=False)

    def insert(self, a):
        self.staged.append(("+", a))
        return True

    def erase(self, a):
        self.staged.append(("-", a))

    def commit(self):
        added = 0
        for op, a in self.staged:
            if op == "+":
                added += 0 if id(a) in self.members else a.nbytes
                self.members[id(a)] = a
            elif self.members.pop(id(a), None) is not None:
                added -= a.nbytes
        self.staged.clear()
        self.commits.append(set(self.members))
        if self.on_commit is not None:
            self.on_commit(added)


class _Vm:
    """macOS memory as a wire draws on it, reported through host_box on
    ``box``: a commit takes free pages down to the kernel's free target
    (vm_page_free_target, 62.5 MiB here), and only then does the kernel
    reclaim file pages for the rest. So free memory sits at the target
    for most of a large wire while the file pages fall. ``grow`` is
    another process taking memory the same way."""

    TARGET = 4000 * 16384

    def __init__(self, box, free, file):
        self.box = box
        self.free, self.file = int(free), int(file)
        self.lowest_free = self.free
        self.publish()

    def publish(self):
        host_box.use(self.box, {
            "free": self.free, "inactive": 0, "purgeable": 0,
            "speculative": 0, "filebacked": self.file,
            "wired": self.box["memory_size"] // 20, "compressor": 0})

    def take(self, nbytes):
        from_free = min(int(nbytes), max(0, self.free - self.TARGET))
        self.free -= from_free
        rest = int(nbytes) - from_free
        reclaimed = min(rest, self.file)
        self.file -= reclaimed
        self.free = max(0, self.free - (rest - reclaimed))
        self.lowest_free = min(self.lowest_free, self.free)
        self.publish()

    grow = take

    def committed(self, nbytes):
        if nbytes > 0:
            self.take(nbytes)

    def read(self):
        return self.free + self.file


def _leaves(n, nbytes):
    """A model of ``n`` arrays of ``nbytes`` each, by size alone."""
    arrays = {f"w{i}": SimpleNamespace(nbytes=int(nbytes), ndim=2)
              for i in range(n)}
    return SimpleNamespace(parameters=lambda: arrays)


def _room(monkeypatch, nbytes, depth=4096):
    room = budget.KvRoom(int(nbytes), depth, 1, 0, 0, 0, priced=True)
    monkeypatch.setattr(budget, "kv_room_bytes", lambda *a, **k: room)
    monkeypatch.setattr(budget, "_WIRE_DEPTH", depth)
    return room


def _wire(m, w, budget_bytes=int(108.2e9)):
    return es._wire_resident_experts(m, None, int(w), budget=budget_bytes,
                                     prefill_step=None, skip_ids=frozenset())


def _stale_mach(monkeypatch):
    """host_statistics64 as a wire inside one of its refreshes reads it
    (about once a second): the counters of now from here on, while the
    sysctl counts move with the box."""
    from gmlx.serve import kernel_vm

    frozen = kernel_vm.snapshot()
    monkeypatch.setattr(kernel_vm, "snapshot", lambda: dict(frozen))


def test_a_full_page_cache_wires_all_of_it(monkeypatch, capsys):
    # A 128 GB Mac with another model's pages cached (free 38.4, speculative
    # 1.6, purgeable 2.3, file-backed 54.6 GiB) and a 70 GiB resident set
    # at the run room: free memory falls to the kernel's target early,
    # and the kernel reclaims file pages for every chunk after that.
    vm = _Vm(host_box.mac(128)[0], free=42.3 * GiB, file=54.6 * GiB)
    kq = _Residency(monkeypatch, on_commit=vm.committed)
    _room(monkeypatch, 8.1 * GiB)
    m = _leaves(70, 1 * GiB)
    assert _wire(m, 70 * GiB)
    assert len(kq.members) == 70 and len(kq.commits) == 35
    assert vm.lowest_free == vm.TARGET           # free sat at the target
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("stale", [False, True])
def test_another_process_growing_mid_wire_stops_it(monkeypatch, capsys,
                                                   stale):
    # The same box: 10 chunks in, another process takes 30 GiB, and
    # reclaimable memory no longer covers the keep and the next chunk.
    # A warm wire commits 2 GiB in 10 to 20 ms, inside one refresh of
    # the mach counters, so only the sysctl counts show the growth.
    vm = _Vm(host_box.mac(128)[0], free=42.3 * GiB, file=54.6 * GiB)
    if stale:
        _stale_mach(monkeypatch)

    def committed(nbytes):
        vm.committed(nbytes)
        if len(kq.commits) == 10:
            vm.grow(30 * GiB)
    kq = _Residency(monkeypatch, on_commit=committed)
    _room(monkeypatch, 8.1 * GiB)
    m = _leaves(60, 1 * GiB)
    assert _wire(m, 60 * GiB) is False
    assert kq.members == {}                      # unwired
    err = capsys.readouterr().err
    assert err.startswith(
        f"[stream] {budget.STREAM_FREE_NOW}: holding them takes 64.4 GB, "
        "and after ")
    assert ("GB the next chunk needs beside the 12.7 GB kept for the "
            "kernel floor and the KV room of 8.7 GB at 4096 tokens (the "
            "run's prompt and max tokens). To keep them in memory, close "
            f"other apps. {budget.OVERCOMMIT_LEVER}") in err
    assert "0.0 GB" not in err


def test_a_live_install_s_wired_bytes_are_not_kept_again(monkeypatch):
    # A 512 GiB Studio serving a second model: the first one's 20 GiB
    # arena is wired, so reclaimable memory leaves it out already, and the
    # wire keeps the floor and the room beside it, not 20 GiB more.
    import gmlx.stream.installs as installs

    monkeypatch.setattr(installs, "live_wired_bytes", lambda: 20 * GiB)
    vm = _Vm(host_box.mac(512)[0], free=200 * GiB, file=100 * GiB)
    kq = _Residency(monkeypatch, on_commit=vm.committed)
    _room(monkeypatch, 8.1 * GiB)
    m = _leaves(250, 1 * GiB)
    assert _wire(m, 250 * GiB, budget_bytes=400 * GiB)
    assert len(kq.members) == 250


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("over", [0, 1 * GiB])
def test_the_wire_stops_where_validate_s_check_does(monkeypatch, over,
                                                     stale):
    # validate's live check and the wire on the same numbers, a 128 GiB
    # Mac with another model's pages cached: a set the check keeps in
    # memory wires in full, and one a GiB past it stops at its last chunk.
    # An app that took 10 GiB a moment ago is in the sysctl counts, and
    # not yet in the mach counters: both read the sysctl counts.
    from gmlx.stream import plan as sp

    box, _ = host_box.mac(128)
    ram = box["memory_size"]
    vm = _Vm(box, free=52.3 * GiB, file=54.6 * GiB)
    if stale:
        _stale_mach(monkeypatch)
    vm.grow(10 * GiB)
    room = sp.box_plan(sp.ModelPlan(
        arch="qwen3moe", total_bytes=GiB, expert_bytes=GiB // 2, groups={},
        moe_layers=48, n_experts=128, experts_per_token=8, ring_bytes=0,
        kv_costs=None, trained_ctx=None), ram_bytes=ram,
        ws_bytes=box["max_recommended_working_set_size"]).room
    keep = budget.wire_keep_bytes(room.bytes)
    w = vm.read() - keep + over
    model = sp.ModelPlan(
        arch="qwen3moe", total_bytes=w, expert_bytes=w * 9 // 10, groups={},
        moe_layers=48, n_experts=128, experts_per_token=8, ring_bytes=0,
        kv_costs=None, trained_ctx=None)
    plan = sp.box_plan(model, ram_bytes=ram,
                       ws_bytes=box["max_recommended_working_set_size"])
    assert plan.verdict == sp.VERDICT_RESIDENT
    check = sp.live_check(model, plan)
    assert check.fits is (not over) and check.keep_bytes == keep
    assert (sp.live_line(check, plan) == sp.LIVE_RESIDENT) is (not over)
    kq = _Residency(monkeypatch, on_commit=vm.committed)
    m = _leaves(w // GiB, GiB)
    m.parameters()["tail"] = SimpleNamespace(nbytes=w % GiB, ndim=2)
    guard = es._ChunkGuard(check.keep_bytes)
    assert es._install_gpu_residency(
        m, {}, include_expert_stacks=True, chunk_bytes=2 * GiB,
        guard=guard) is (not over)
    wired = sum(a.nbytes for a in kq.members.values())
    # It wires up to the check's line, and refuses the chunk that would
    # cross it.
    assert wired == w if not over else wired <= w - over < wired + guard.pending


def test_without_the_sysctl_counts_the_check_and_the_guard_read_mach(
        monkeypatch):
    # A sysctl read that fails: both read host_statistics64, as the
    # governor does.
    from gmlx.serve import kernel_vm

    _box(host_box.mac(128)[0], rec=100 * GiB, fast=90 * GiB)
    monkeypatch.setattr(kernel_vm, "live_reclaimable_bytes", lambda: None)
    check = _check(99 * GiB)
    assert check.reclaimable_bytes == budget.reclaimable_ram_bytes()
    assert check.reason == "reclaimable"     # 99 GiB, 100 less the keep
    guard = es._ChunkGuard(check.keep_bytes)
    assert guard(2 * GiB) and guard.last == 100 * GiB


@pytest.mark.skipif(sys.platform != "darwin", reason="the sysctl counts")
def test_the_live_read_counts_the_speculative_pages_once(monkeypatch):
    # sysctl's free count leaves out the speculative pages, and its
    # pageable external count holds them.
    from gmlx.serve import kernel_vm

    counts = {"vm.page_free_count": 2_326_173,
              "vm.page_speculative_count": 216_071,
              "vm.page_purgeable_count": 48_710,
              "vm.page_pageable_external_count": 4_093_334}
    monkeypatch.setattr(kernel_vm, "_sysctl_count", counts.get)
    live = host_box.real("live_reclaimable")
    assert live() == (2_326_173 + 48_710 + 4_093_334) * kernel_vm._PAGE
    del counts["vm.page_purgeable_count"]
    assert live() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="the sysctl counts")
def test_the_live_read_matches_this_mac_s_mach_counters():
    # A misnamed count would leave every wire on the fallback.
    real = host_box.real("snapshot")()
    live = host_box.real("live_reclaimable")()
    assert live is not None
    # Mach's free and file-backed counts both hold the speculative pages.
    want = (real["free"] + real["purgeable"] + real["filebacked"]
            - real["speculative"])
    assert abs(live - want) <= max(2 * GiB, want // 10)


def test_the_first_chunk_under_the_line_never_commits(monkeypatch):
    vm = _Vm(host_box.mac(128)[0], free=5 * GiB, file=0)
    kq = _Residency(monkeypatch, on_commit=vm.committed)
    m = _leaves(4, 1 * GiB)
    assert es._install_gpu_residency(
        m, {}, include_expert_stacks=True, chunk_bytes=2 * GiB,
        guard=es._ChunkGuard(4 * GiB)) is False
    assert kq.commits == []
    es._unwire(m)
    assert kq.commits == [set()]


def test_the_last_partial_chunk_is_guarded_too(monkeypatch):
    # 3 GiB in 2 GiB chunks beside a 4 GiB keep: the first finds 6.5 GiB
    # over its 6 GiB line, the 1 GiB left finds 4.5 GiB under its 5.
    vm = _Vm(host_box.mac(128)[0], free=6.5 * GiB, file=0)
    kq = _Residency(monkeypatch, on_commit=vm.committed)
    m = _leaves(3, 1 * GiB)
    guard = es._ChunkGuard(4 * GiB)
    assert es._install_gpu_residency(
        m, {}, include_expert_stacks=True, chunk_bytes=2 * GiB,
        guard=guard) is False
    assert len(kq.commits) == 1 and len(kq.members) == 2
    assert guard.line == 5 * GiB and guard.last == 4.5 * GiB
    assert es._unwire(m) == 3 * GiB and kq.members == {}


def _reads(monkeypatch, *values):
    """budget.live_reclaimable_ram_bytes reports ``values`` in turn,
    then the last."""
    seq = [int(v) for v in values]

    def read():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    monkeypatch.setattr(budget, "live_reclaimable_ram_bytes", read)


def test_the_guard_waits_while_reclaimable_memory_rises(monkeypatch):
    # Another process frees memory: 1 GiB a step until the line.
    _reads(monkeypatch, 1 * GiB, 2 * GiB, 3 * GiB, 4 * GiB, 9 * GiB)
    assert es._ChunkGuard(4 * GiB)(2 * GiB)


def test_a_rise_under_the_minimum_does_not_hold_the_wait(monkeypatch):
    # 8 MiB a step (160 MiB a second) keeps rising, but never by the
    # 256 MiB that counts within the stall window, so the wait ends there.
    seq = [1 * GiB + i * (8 << 20) for i in range(1000)]
    monkeypatch.setattr(budget, "live_reclaimable_ram_bytes",
                        lambda: seq.pop(0))
    guard = es._ChunkGuard(4 * GiB)
    assert not guard(2 * GiB)
    assert guard.waited == pytest.approx(es._WIRE_STALL_S, abs=0.2)


def test_the_guard_gives_up_once_reclaimable_memory_stops_rising(monkeypatch):
    _reads(monkeypatch, 1 * GiB, 1 * GiB, 9 * GiB)
    assert not es._ChunkGuard(4 * GiB, stall_s=0.0)(2 * GiB)


def test_the_waits_of_one_wire_stop_at_the_deadline(monkeypatch):
    # Memory that keeps rising by 512 MiB a step from far under the line:
    # every rise counts, and the wait ends at the wire's deadline.
    level = {"v": 0}

    def read():
        level["v"] += 512 << 20
        return level["v"]
    monkeypatch.setattr(budget, "live_reclaimable_ram_bytes", read)
    guard = es._ChunkGuard(1 << 50)
    assert not guard(2 * GiB)
    assert guard.waited == pytest.approx(es._WIRE_WAIT_S, abs=0.1)
    # A later chunk of the same wire has no wait left.
    assert not guard(2 * GiB)
    assert guard.waited < es._WIRE_WAIT_S + 0.1


# --- the table-only install ----------------------------------------------


def _table_only(monkeypatch, fits=False):
    """A qwen4exp-like model over the budget whose table alone brings it
    under, with the wire chunk at one byte so every array is a chunk,
    and no exemption, so its small wire is checked and guarded.
    ``fits``: under the budget, its table streamed by GMLX_STREAM_PLE=1."""
    from mlx.utils import tree_flatten

    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_table_stream import _moe_table_model

    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "1")
    monkeypatch.setenv("GMLX_GPU_KEEPWARM", "0")
    monkeypatch.setenv("GMLX_WIRE_CHUNK_GB", str(1 / GiB))
    monkeypatch.setattr(budget, "WIRE_EXEMPT_BYTES", 0)
    model, glu = _moe_table_model()
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    ws = (total - ts.table_bytes(model) // 2) / 0.9
    if fits:
        monkeypatch.setenv("GMLX_STREAM_PLE", "1")
        ws = 2 * total / 0.9
    # The box's own device info (its memory size) under the test budget.
    real = mx.device_info
    monkeypatch.setattr(mx, "device_info", lambda *a, **k: {
        **real(*a, **k), "max_recommended_working_set_size": ws})
    experts = {id(getattr(glu, p).weight)
               for p in ("gate_proj", "up_proj", "down_proj")}
    return model, glu, experts


def test_a_table_only_install_wires_the_experts_when_they_fit(monkeypatch):
    kq = _Residency(monkeypatch)
    model, glu, experts = _table_only(monkeypatch)
    box, _ = host_box.mac(128)
    _box(box, rec=100 * GiB, fast=90 * GiB)
    install_expert_streaming(model)
    assert ts.table_streaming_active(model)
    assert not getattr(glu, "_kq_cpu_only", False)
    assert experts <= set(kq.members)


def test_a_table_only_install_streams_the_experts_when_they_do_not_fit(
        monkeypatch, capsys):
    kq = _Residency(monkeypatch)
    model, glu, experts = _table_only(monkeypatch)
    box, _ = host_box.runner()
    _box(box, rec=0.5 * GiB, fast=0.5 * GiB)
    install_expert_streaming(model)
    assert ts.table_streaming_active(model)
    assert getattr(glu, "_kq_cpu_only", False)
    assert not any(experts & c for c in kq.commits)
    err = capsys.readouterr().err
    assert f"[stream] {budget.STREAM_FREE_NOW}: holding them takes " in err
    assert "of reclaimable memory less" in err
    assert err.rstrip().endswith(budget.OVERCOMMIT_LEVER)


def test_the_ci_runner_wires_a_tiny_table_model_s_experts(monkeypatch):
    # The runner's reclaimable memory is under the keep, and a wire of one
    # chunk is judged on the budget alone, by the check and the wire both.
    kq = _Residency(monkeypatch)
    model, glu, experts = _table_only(monkeypatch)
    monkeypatch.delenv("GMLX_WIRE_CHUNK_GB")
    monkeypatch.setattr(budget, "WIRE_EXEMPT_BYTES", 2 << 30)
    box, vm = host_box.runner()
    host_box.use(box, vm)
    assert budget.live_reclaimable_ram_bytes() < budget.ram_bytes() // 5
    install_expert_streaming(model)
    assert not getattr(glu, "_kq_cpu_only", False)
    assert experts <= set(kq.members)


@pytest.mark.parametrize("fits", [False, True])
def test_reclaimable_memory_that_stalls_mid_wire_unwires_and_streams(
        monkeypatch, capsys, fits):
    import gmlx.load.loadlog as loadlog

    monkeypatch.setattr(loadlog, "_LAST_VERBOSE", True)
    kq = _Residency(monkeypatch)
    model, glu, experts = _table_only(monkeypatch, fits=fits)
    box, _ = host_box.mac(128)
    _box(box, rec=100 * GiB, fast=90 * GiB)
    # The check and the first chunk find 100 GiB; then 1 GiB.
    _reads(monkeypatch, 100 * GiB, 100 * GiB, 1 * GiB)
    install_expert_streaming(model)
    assert getattr(glu, "_kq_cpu_only", False)
    assert len(kq.commits) > 1 and not experts & set(kq.members)
    out, err = capsys.readouterr()
    assert f"[stream] {budget.STREAM_FREE_NOW}: holding them takes " in err
    assert " of it wired, reclaimable memory stopped at 1.1 GB, under the " in err
    # The install's own lines name the fallback, not a flag.
    assert "its experts do not fit the memory free now" in out + err
    assert "even though" not in out + err


@pytest.mark.parametrize("above", [True, False])
def test_overcommit_keeps_the_guard_on_at_the_kernel_floor(
        monkeypatch, capsys, above):
    # A 16 GiB Mac whose reclaimable memory sits half a GiB over or under
    # the 1.7 GB kernel floor, under the 3.2 GiB keep: overcommit skips the
    # check, and each chunk keeps only the floor.
    kq = _Residency(monkeypatch)
    model, glu, experts = _table_only(monkeypatch)
    box, _ = host_box.mac(16)
    _box(box, rec=14 * GiB, fast=10 * GiB)
    floor = budget.kernel_floor_bytes()
    assert floor == 0.1 * box["memory_size"]
    rec = floor + (GiB // 2 if above else -GiB // 2)
    _reads(monkeypatch, rec)
    assert not budget.check_resident_wire(
        GiB, room_bytes=0).fits                    # without overcommit
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    install_expert_streaming(model)
    assert getattr(glu, "_kq_cpu_only", False) is not above
    if above:
        assert experts <= set(kq.members)
    else:
        assert not any(experts & c for c in kq.commits)
        err = capsys.readouterr().err
        assert err.rstrip().endswith(
            "the next chunk needs beside the 1.7 GB kernel floor. To keep "
            "them in memory, close other apps.")
