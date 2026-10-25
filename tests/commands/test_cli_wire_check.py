#!/usr/bin/env python3
"""`gmlx run` and `gmlx chat` judge a MoE's resident set against live
memory before the weights load (cli.preload_wire_check): one that does
not fit streams its experts. Header sizes and memory counters are
scripted."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import host_box
import gmlx.commands.cli as cli
import gmlx.stream.budget as budget
import gmlx.stream.table_stream as ts

GiB = 1 << 30


def _box(box, *, unwired):
    """``box`` with ``unwired`` bytes neither wired nor compressed."""
    host_box.use(box, {
        "free": int(unwired), "inactive": 0, "purgeable": 0,
        "speculative": 0, "filebacked": 0,
        "wired": box["memory_size"] - int(unwired), "compressor": 0})


@pytest.fixture(autouse=True)
def _fresh_table_latch(monkeypatch):
    import gmlx.serve.capacity as cap

    monkeypatch.setattr(ts, "_FOR_MEMORY", set())
    monkeypatch.setattr(ts, "_GUARDED", {})
    # A short check settles on scripted counters without a wait.
    monkeypatch.setattr(cap, "_SETTLE_STEP_S", 0)


@pytest.fixture
def header(monkeypatch, tmp_path):
    """A GGUF path whose header reports ``text`` bytes, ``experts`` of
    them routed experts, ``tables`` of streamable tables, ``deferred``
    read from the file, under a stream budget of ``budget_bytes``, with a
    KV room of ``room`` bytes."""
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    for k in ("GMLX_OVERCOMMIT", "GMLX_WIRE_CHUNK_GB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(budget, "_WIRE_DEPTH", None)

    def set_header(text, experts=0, deferred=0, budget_bytes=None,
                   room=1 * GiB, tables=0):
        import gmlx.device as device
        import gmlx.load.discovery as discovery
        import gmlx.serve.capacity as cap

        monkeypatch.setattr(discovery, "text_tensor_bytes",
                            lambda p: int(text))
        monkeypatch.setattr(cap, "streamed_expert_bytes",
                            lambda p: int(experts))
        monkeypatch.setattr(cap, "deferred_table_bytes",
                            lambda p, *a, **k: int(deferred))
        monkeypatch.setattr(cap, "streamed_table_bytes",
                            lambda p: int(tables))
        monkeypatch.setattr(device, "stream_budget_bytes",
                            lambda: budget_bytes)
        monkeypatch.setattr(
            budget, "kv_room_bytes",
            lambda p, env=None, step=None, depth=None, resident=False: budget.KvRoom(
                int(room), depth or 32768, 1, int(room), 0, 0, priced=True))
        return str(f)
    return set_header


def _args(path, **kw):
    base = dict(gguf=path, stream_experts=False, stream_cpu=False,
                mmproj=None, draft_gguf=None, prefill_step_size=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_moe_that_does_not_fit_streams_its_experts(header, capsys):
    # Flash-Next's 82.5 GB resident set on a 128 GB Mac whose other
    # processes leave 85 GiB unwired, priced at a run's small room.
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9),
                  room=8.1 * GiB)
    _box(host_box.mac(128)[0], unwired=85 * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert args.stream_experts is True
    err = capsys.readouterr().err
    assert err.startswith(f"[stream] {budget.STREAM_FREE_NOW}: holding them "
                          "takes 82.5 GB, against 91.3 GB of memory not wired")
    assert ("less 12.7 GB kept for the kernel floor and the KV room of "
            "8.7 GB at 4096 tokens (the run's prompt and max tokens)") in err
    assert err.rstrip().endswith(
        f"quit another model or close other apps. {budget.OVERCOMMIT_LEVER}")
    assert budget.wire_room_depth() == 4096
    assert not ts.resident_wire_guarded(path)


def test_a_run_started_as_the_last_one_exits_keeps_its_experts(
        header, monkeypatch):
    # The same set and box, read while the run before it exits: its pages
    # unwire over a few seconds, to 105 GiB not wired.
    import gmlx.serve.capacity as cap

    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9),
                  room=8.1 * GiB)
    box = host_box.mac(128)[0]
    _box(box, unwired=85 * GiB)

    def settle(sample, free, need):
        _box(box, unwired=105 * GiB)
        return sample()

    monkeypatch.setattr(cap, "_settle_unwired", settle)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert not args.stream_experts
    assert ts.resident_wire_guarded(path)


@pytest.mark.parametrize("budget_bytes,unwired,mtp_off", [
    (108.2e9, 89 * GiB, False), (85e9, 85 * GiB, True)])
def test_a_run_started_as_the_last_one_exits_keeps_what_fits(
        header, monkeypatch, budget_bytes, unwired, mtp_off):
    # The same set beside its 4 GB auto drafter, read while the run
    # before it exits and at 105 GiB once it has. Under the 108.2 GB
    # budget the drafter fits beside the experts. Under 85 GB only the
    # experts do, and at 85 GiB not wired they did not yet.
    import gmlx.load.discovery as discovery
    import gmlx.serve.capacity as cap

    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: True)
    monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
    monkeypatch.setattr(discovery, "mtp_block_bytes", lambda p: int(4e9))
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(budget_bytes),
                  room=8.1 * GiB)
    box = host_box.mac(128)[0]
    _box(box, unwired=unwired)

    def settle(sample, free, need):
        _box(box, unwired=105 * GiB)
        return sample()

    monkeypatch.setattr(cap, "_settle_unwired", settle)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert not args.stream_experts
    assert bool(getattr(args, "no_speculative", False)) is mtp_off


@pytest.mark.parametrize("drafter,asked,steps", [
    (0, None, {2048}), (0, 1024, {1024}), (4e9, None, {4096, 2048})])
def test_a_run_prices_the_prefill_chunk_it_takes(
        header, monkeypatch, drafter, asked, steps):
    # A run prefills a model kept in memory in chunks of the asked step or
    # 2048 rows, and of 4096 beside an MTP drafter, whatever step serve
    # takes for the arch.
    import gmlx.load.discovery as discovery

    if drafter:
        monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: True)
        monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
        monkeypatch.setattr(discovery, "mtp_block_bytes",
                            lambda p: int(drafter))
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9),
                  room=8.1 * GiB)
    seen = set()
    priced = budget.kv_room_bytes

    def kv_room_bytes(p, step=None, **k):
        seen.add(step)
        return priced(p, step=step, **k)

    monkeypatch.setattr(budget, "kv_room_bytes", kv_room_bytes)
    _box(host_box.mac(128)[0], unwired=120 * GiB)
    cli.preload_wire_check(_args(path, prefill_step_size=asked), depth=4096)
    assert seen == steps


def test_a_run_whose_tables_must_go_to_the_file_settles_too(
        header, monkeypatch):
    # A 100 GB set over its 85 GB budget, which its 20 GB of tables read
    # from the file bring under, read while the run before it exits.
    import gmlx.serve.capacity as cap

    path = header(100e9, experts=77.0e9, tables=20e9, budget_bytes=int(85e9),
                  room=8.1 * GiB)
    box = host_box.mac(128)[0]
    _box(box, unwired=85 * GiB)

    def settle(sample, free, need):
        _box(box, unwired=105 * GiB)
        return sample()

    monkeypatch.setattr(cap, "_settle_unwired", settle)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert not args.stream_experts
    assert ts.tables_for_memory(path)


@pytest.mark.parametrize("drafter", [0, 4e9])
def test_a_smaller_file_of_a_moe_never_keeps_less_in_memory(
        header, monkeypatch, drafter):
    # Flash-Next's quants on a 128 GB Mac with 104.3 GB unwired, at
    # the serve room, alone or beside an auto MTP drafter. Each holds the
    # same 28.8 GB per-layer table, which reads from the file before the
    # experts or the drafter of a smaller file go. 93.7 GB is UD-IQ4_XS,
    # 111.3 GB UD-Q4_K_XL.
    import gmlx.load.discovery as discovery

    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: bool(drafter))
    monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
    monkeypatch.setattr(discovery, "mtp_block_bytes", lambda p: int(drafter))
    table, limit = 28.8e9, 108.2e9
    sizes = (60e9, 80e9, 90e9, 93.7e9, 100e9, 105e9, 111.3e9, 120e9, 130e9,
             140e9)
    kept, latched = [], []
    for text in sizes:
        deferred = table if text > limit >= text - table else 0
        path = header(text, experts=0.64 * text, deferred=deferred,
                      tables=table, budget_bytes=int(limit), room=14.4e9)
        _box(host_box.mac(128)[0], unwired=104.3e9)
        args = _args(path)
        cli.preload_wire_check(args, depth=None)
        # 2: the experts and any drafter in memory; 1: the experts.
        kept.append(0 if args.stream_experts
                    else 1 if getattr(args, "no_speculative", False) else 2)
        latched.append(ts.tables_for_memory(path))
    assert kept == [2] * (7 if not drafter else 6) + [1] * bool(drafter) + [0] * 3
    # The table step of 90 to 105 GB; UD-Q4_K_XL's budget step needs none.
    assert latched == [text in (90e9, 93.7e9, 100e9, 105e9) for text in sizes]


# 110 GiB: an idle box. 94.7 GiB: 15.3 GiB more wired by other processes.
@pytest.mark.parametrize("unwired", [110, 94.7])
def test_a_128gb_box_keeps_flash_next_resident_at_the_run_room(header, capsys,
                                                               unwired):
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9),
                  room=8.1 * GiB)
    _box(host_box.mac(128)[0], unwired=unwired * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert args.stream_experts is False
    assert capsys.readouterr() == ("", "")
    # The warm touch leaves the experts to the install's chunked wire.
    assert ts.resident_wire_guarded(path)


def test_a_run_keeps_the_room_of_its_own_prefill_beside_the_experts(
        header, monkeypatch):
    # Flash-Next kept in memory prefills at its resident step, so the run
    # keeps 4 GiB of KV room, not the 14 GiB of a streamed chunk, and its
    # install's guard keeps the same room.
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9))
    rooms = {True: 4 * GiB, False: 14 * GiB}
    monkeypatch.setattr(
        budget, "kv_room_bytes",
        lambda p, env=None, step=None, depth=None, resident=False:
        budget.KvRoom(rooms[resident], depth or 32768, 1, rooms[resident], 0,
                      0, priced=True))
    box = host_box.mac(128)[0]
    _box(box, unwired=0)
    keep = budget.wire_keep_bytes(4 * GiB)
    _box(box, unwired=82.5e9 + keep + GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=4096)
    assert args.stream_experts is False
    assert ts.guarded_wire_keep(path)[0].bytes == 4 * GiB


@pytest.mark.parametrize("text,deferred,tail", [
    # Flash-Next with its table held: past the budget, within RAM.
    (111.3e9, 0, f". {budget.OVERCOMMIT_LEVER}"),
    # DS4.1's bare path on a 128 GB Mac: no lever holds 165.3 GB.
    (300.5e9, 135.2e9, " and more than RAM."),
])
def test_a_moe_past_the_stream_budget_streams(header, capsys, text,
                                              deferred, tail):
    # Over the budget, whatever the box has free.
    path = header(text, experts=77.0e9, deferred=deferred,
                  budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=1000 * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True
    assert capsys.readouterr().err == (
        f"[stream] {budget.STREAM_OVER_BUDGET}: holding them takes "
        f"{(text - deferred) / 1e9:.1f} GB, past the 108.2 GB budget{tail}\n")


@pytest.mark.parametrize("text,experts,deferred,tables", [
    (300.5e9, 156.9e9, 135.2e9, 0),
    # A 150 GB Flash-Next quant, over the budget without its table too.
    (150e9, 100e9, 0, 28.8e9),
])
def test_overcommit_streams_a_set_larger_than_ram(header, monkeypatch,
                                                  capsys, text, experts,
                                                  deferred, tables):
    # DS4.1's bare path on a 128 GB Mac: GMLX_OVERCOMMIT=1 keeps no set
    # larger than RAM, so its experts stream as without it.
    path = header(text, experts=experts, deferred=deferred, tables=tables,
                  budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=120 * GiB)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True
    assert not ts.tables_for_memory(path)
    # The table streams from the file with the experts.
    head = (f"[stream] lookup tables of {tables / 1e9:.1f} GB offloaded to "
            "disk\n" if tables else "")
    assert capsys.readouterr().err == head + (
        f"[stream] {budget.STREAM_OVER_BUDGET}: holding them takes "
        f"{(text - deferred) / 1e9:.1f} GB, past the 108.2 GB budget and "
        "more than RAM.\n")


def test_the_tables_a_load_reads_from_the_file_are_not_wired(header):
    path = header(300.5e9, experts=156.9e9, deferred=200e9,
                  budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=120 * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False


def test_a_dense_model_loads_as_on_main(header, capsys):
    # An 8B Q8 dense model on a 16 GB Mac with apps open: no check, no
    # line, no placement change.
    path = header(8.5e9, experts=0)
    _box(host_box.mac(16)[0], unwired=9 * GiB)
    ts.guard_resident_wire(path)             # a MoE's load of this path
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False
    assert capsys.readouterr() == ("", "")
    assert not ts.resident_wire_guarded(path)


def test_the_ci_runner_loads_a_tiny_moe(header):
    path = header(10e6, experts=8e6)
    box, vm = host_box.runner()
    host_box.use(box, vm)
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False


@pytest.mark.parametrize("asked", ["stream_experts", "stream_cpu"])
def test_a_streamed_placement_is_left_alone(header, asked):
    path = header(165.3e9, experts=156.9e9, budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=10 * GiB)
    ts.guard_resident_wire(path)             # a resident load of this path
    args = _args(path, **{asked: True})
    cli.preload_wire_check(args, depth=2048)
    assert args.stream_experts is (asked == "stream_experts")
    assert not ts.resident_wire_guarded(path)
    # The install's own check prices the run's room too.
    assert budget.wire_room_depth() == 2048


def test_overcommit_keeps_the_experts_resident(header, monkeypatch):
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=50 * GiB)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False


def test_overcommit_gets_the_16gb_example_past_the_fit_check(header,
                                                            monkeypatch,
                                                            capsys):
    # gpt-oss-20b-mxfp4 (12.1 GB of text tensors, 10.2 GB of them experts)
    # on a 16 GB Mac, whose 11.45 GB working set gives a 10.3 GB budget.
    # The line offers GMLX_OVERCOMMIT=1, and under it the run's fit check
    # judges the 12.1 GB file against the 16 GB of RAM, not the working
    # set, and lets the resident load through.
    import os

    import gmlx.commands.tool_preflight as tp

    monkeypatch.delenv("GMLX_TOOL_PREFLIGHT", raising=False)
    ws = 11_453_251_584
    path = header(12_096_558_336, experts=10_152_345_600,
                  budget_bytes=int(0.9 * ws))
    _box(host_box.mac(16)[0], unwired=12 * GiB)
    size = os.path.getsize
    monkeypatch.setattr(tp, "_shards", lambda p: [p])
    monkeypatch.setattr(tp.os.path, "getsize",
                        lambda p: 12_109_566_560 if p == path else size(p))
    monkeypatch.setattr(tp, "_synth_config", lambda p: {
        "num_hidden_layers": 24, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 64,
        "max_position_embeddings": 131072})
    monkeypatch.setattr(tp, "working_set_bytes", lambda: ws)
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True
    assert capsys.readouterr().err.rstrip().endswith(budget.OVERCOMMIT_LEVER)
    assert tp.check_or_exit(path, streaming=args.stream_experts) is None
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False
    assert tp.check_or_exit(path, streaming=args.stream_experts)["fits"]
    monkeypatch.delenv("GMLX_OVERCOMMIT")
    with pytest.raises(SystemExit):
        tp.check_or_exit(path)
    monkeypatch.setenv("GMLX_TOOL_PREFLIGHT", "0")
    assert tp.check_or_exit(path) is None


def _tiny_moe():
    """A one-layer MoE of a few KB with q8_0 experts, as a run's load hands
    it to the install."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "load"))
    from test_offload import _holder_model, _kquant_glu

    glu = _kquant_glu()
    model = _holder_model(glu)
    model.parameters = lambda: {"glu": glu.parameters()}
    return model


class _Feeder:
    """A built feeder, prefill ring or decode arena, as the install sees
    it: it covers every layer and holds no memory."""

    n_slots, slot_bytes, nominal_bytes, locked_bytes = 2, 1, 0, 0
    _mlock_deferred, _fast_disk, _probe_bps = True, False, 0.0

    def covers(self, li):
        return True

    def release_slots(self):
        pass

    lend_for_ring = seed_from_ring = arena_sources = release_slots


def _build_feeders(monkeypatch):
    """The run's install builds both feeders, as it does on a real file."""
    from types import SimpleNamespace

    import gmlx.stream.decode_feeder as dfm
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.prefetch as pf
    import gmlx.stream.prefill_feeder as pff

    offsets = {0: [("m.gguf", k, 1, 4, kind)
                   for k, kind in enumerate(("gate", "up", "down"))]}
    monkeypatch.setattr(pf, "maybe_make_prefetcher",
                        lambda path: SimpleNamespace(offsets=offsets))
    monkeypatch.setattr(es, "_prefill_ring_reason", lambda *a: None)
    monkeypatch.setattr(pff, "maybe_make_prefill_feeder",
                        lambda *a, **k: _Feeder())
    monkeypatch.setattr(dfm, "maybe_make_decode_feeder",
                        lambda *a, **k: _Feeder())
    for k in ("GMLX_STREAM_UNMAP_STACKS", "GMLX_GPU_KEEPWARM",
              "GMLX_DECODE_LOOKAHEAD"):
        monkeypatch.setenv(k, "0")
    monkeypatch.setenv("GMLX_FEEDER_DECODE", "1")
    monkeypatch.delenv("GMLX_GPU_AUTONOMOUS", raising=False)


@pytest.mark.parametrize("feeders", [False, True])
@pytest.mark.parametrize("leg", ["unwired", "budget"])
def test_a_run_its_check_streams_names_the_check_not_a_flag(
        header, monkeypatch, capsys, tmp_path, leg, feeders):
    # The run's check streams the experts of a MoE under the stream
    # budget: on a busy Mac (85 GiB unwired), or with a 2.8 GB drafter
    # that takes the set past the budget. The install's lines say why, and
    # none says that the flag streams a model that fits, with its feeders
    # built or not.
    import importlib

    import mlx.core as mx

    import gmlx.load.loadlog as loadlog

    if leg == "unwired":
        path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9),
                      room=8.1 * GiB)
        _box(host_box.mac(128)[0], unwired=85 * GiB)
        args = _args(path)
    else:
        path = header(106e9, experts=95e9, budget_bytes=int(108.2e9))
        _box(host_box.mac(128)[0], unwired=120 * GiB)
        drafter = tmp_path / "drafter.gguf"
        with open(drafter, "wb") as f:
            f.truncate(int(2.8e9))
        args = _args(path, draft_gguf=str(drafter))
    cli.preload_wire_check(args, depth=4096)
    assert args.stream_experts is True and args.stream_fallback == leg
    capsys.readouterr()
    monkeypatch.setattr(loadlog, "_LAST_VERBOSE", True)
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "0")
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    # The package re-exports generate(), so import the module by name.
    gen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
    monkeypatch.setattr(gen, "wired_limit", gen.wired_limit)
    args.gguf = None                   # the tiny model has no file
    if feeders:
        _build_feeders(monkeypatch)
    cli._apply_placement(args, _tiny_moe())
    out, err = capsys.readouterr()
    why = {"unwired": "its experts do not fit the memory free now",
           "budget": "its experts do not fit the memory budget"}[leg]
    head = [ln for ln in (out + err).splitlines()
            if ln.startswith("[stream] streaming: ")]
    assert len(head) == 1 and head[0].startswith(
        f"[stream] streaming: model 0 GB, {why} - 1 MoE layers' experts "
        "(0.0 GB) stay file-backed; rest of the model + KV on ")
    assert "even though" not in out + err
    assert ("decode feeder: 0.0 GB popularity-managed expert arena"
            in out + err) is feeders


def test_a_chat_on_a_vlm_has_no_streamed_placement(header):
    path = header(82.5e9, experts=77.0e9, budget_bytes=int(108.2e9))
    _box(host_box.mac(128)[0], unwired=50 * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=None, can_stream=False)
    assert args.stream_experts is False


@pytest.fixture
def glm_head(header, monkeypatch):
    """A 70 GiB trunk whose own 2.79 GB NextN block auto-MTP loads beside
    it (GLM-5.3), with no KV room: the 4 GB kernel floor kept."""
    import gmlx.load.discovery as discovery

    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: True)
    monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
    monkeypatch.setattr(discovery, "mtp_block_bytes", lambda p: int(2.79e9))
    return header(70 * GiB, experts=60 * GiB, budget_bytes=200 * GiB,
                  room=0)


def test_an_auto_head_that_fits_beside_the_experts_loads(glm_head, capsys):
    _box(host_box.mac(128)[0], unwired=80 * GiB)
    args = _args(glm_head)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False      # 72.6 <= 76.3
    assert not getattr(args, "no_speculative", False)
    assert capsys.readouterr() == ("", "")


def test_an_auto_head_that_tips_the_experts_over_is_dropped(glm_head,
                                                             capsys):
    # Resident with no MTP decodes faster than streamed with no MTP, and
    # a streamed load takes no auto drafter.
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    args = _args(glm_head)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False      # 72.6 > 71.3, 70 <= 71.3
    assert args.no_speculative is True
    assert cli.resolve_speculative(args, glm_head) == (False, "")
    assert capsys.readouterr().err == (
        "[mtp] auto MTP is off: the experts stay in memory, and the 2.8 GB "
        "MTP drafter does not fit beside them. --speculative streams the "
        "experts to keep it.\n")


def test_experts_that_do_not_fit_without_the_auto_head_stream(glm_head,
                                                              capsys):
    _box(host_box.mac(128)[0], unwired=69 * GiB)
    args = _args(glm_head)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True       # 70 > 65.3
    assert not getattr(args, "no_speculative", False)
    assert cli.resolve_speculative(args, glm_head) == (False, "")
    # The streamed run takes no auto head, and the line counts the model
    # alone.
    err = capsys.readouterr().err
    assert err.startswith(f"[stream] {budget.STREAM_FREE_NOW}: holding them "
                          "takes 75.2 GB, against 74.1 GB of memory not wired")


@pytest.mark.parametrize("unwired,streams", [(75, False), (69, True)])
def test_a_config_drafter_file_marked_auto_goes_as_a_head_does(
        header, tmp_path, unwired, streams):
    # An entry's draft_gguf under `speculative: auto` loads only when the
    # experts fit in memory beside it.
    path = header(70 * GiB, experts=60 * GiB, budget_bytes=200 * GiB, room=0)
    drafter = tmp_path / "dflash-m.gguf"
    with open(drafter, "wb") as f:
        f.truncate(int(2.79e9))
    _box(host_box.mac(128)[0], unwired=unwired * GiB)
    args = _args(path, draft_gguf=str(drafter), draft_gguf_auto=True)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is streams and args.draft_gguf is None
    assert cli.resolve_speculative(args, path) == (False, "")


def test_a_vlm_that_streams_counts_the_head_it_keeps(glm_head, tmp_path,
                                                     capsys):
    # A text-only VLM run keeps its MTP drafter when its experts stream.
    mmproj = tmp_path / "mmproj.gguf"
    with open(mmproj, "wb") as f:
        f.truncate(GiB // 2)
    _box(host_box.mac(128)[0], unwired=69 * GiB)
    args = _args(glm_head, mmproj=str(mmproj), image=None, audio=None)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True       # 70.5 > 65.3
    assert not getattr(args, "no_speculative", False)
    assert cli._run_mode(args) == "vlm_mtp"
    err = capsys.readouterr().err
    assert err.startswith(f"[stream] {budget.STREAM_FREE_NOW}: holding them "
                          "takes 78.5 GB, against 74.1 GB of memory not wired")


def test_a_dropped_head_prices_the_room_of_the_run_without_it(
        glm_head, monkeypatch, capsys):
    # A capped run that would load the auto head prices the serve context
    # (6 GiB of KV here); without the head it prices its own 4096 tokens
    # (1 GiB). Keeps of 9.7 and 4.7 GiB against 77 GiB: the set fits only
    # at the run's room.
    rooms = {None: 6 * GiB, 4096: 1 * GiB}
    monkeypatch.setattr(
        budget, "kv_room_bytes",
        lambda p, env=None, step=None, depth=None, resident=False: budget.KvRoom(
            rooms[depth], depth or 32768, 1, rooms[depth], 0, 0, priced=True))
    _box(host_box.mac(128)[0], unwired=77 * GiB)
    serve = budget.check_resident_wire(70 * GiB, room_bytes=6 * GiB)
    assert not serve.fits
    args = _args(glm_head)
    cli.preload_wire_check(args, depth=None, depth_without_mtp=4096)
    assert args.stream_experts is False      # 70 <= 77 - 4.7
    assert args.no_speculative is True
    assert budget.wire_room_depth() == 4096
    assert capsys.readouterr().err.startswith("[mtp] auto MTP is off")


@pytest.mark.parametrize("flag", [dict(speculative=True),
                                  dict(native_mtp=True)])
def test_a_head_a_flag_asks_for_streams_the_experts_beside_it(
        glm_head, flag):
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    args = _args(glm_head, **flag)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True       # 72.6 > 71.3
    assert not getattr(args, "no_speculative", False)
    assert cli.resolve_speculative(args, glm_head)[0] is True


@pytest.mark.parametrize("image", ["", "cat.png"])
def test_a_vlm_run_counts_the_head_only_for_a_text_only_run(glm_head,
                                                             tmp_path, image):
    # A --mmproj run with an image decodes on the plain VLM path.
    mmproj = tmp_path / "mmproj.gguf"
    with open(mmproj, "wb") as f:
        f.truncate(GiB // 2)
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    args = _args(glm_head, mmproj=str(mmproj), image=image, audio=None)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False      # 70.5 <= 71.3
    assert getattr(args, "no_speculative", False) is (not image)


def test_a_bench_depths_drafter_counts_in_the_wire(header, tmp_path):
    # --draft-gguf alone runs the --bench-depths MTP A/B: the run sets
    # --speculative from it only after this check.
    path = header(70 * GiB, experts=60 * GiB, budget_bytes=200 * GiB,
                  room=0)
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    draft = tmp_path / "draft.gguf"
    with open(draft, "wb") as f:
        f.truncate(2 * GiB)
    args = _args(path, draft_gguf=str(draft), bench_depths="0,8192")
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True       # 72 > 71.3


@pytest.mark.parametrize("fits,whole", [(True, False), (False, False),
                                        (True, True)])
def test_a_table_gathered_per_layer_that_brings_the_model_under_is_not_wired(
        monkeypatch, tmp_path, capsys, fits, whole):
    # Flash-Next over the budget by its per-layer embedding table alone:
    # the load serves the table from the file and judges the rest, where
    # its text bytes are past the budget. Its install wires the experts in
    # chunks, which GMLX_OVERCOMMIT=1 still stops. The minted headers are
    # small, so the exemption is off. ``whole``: the whole file under the
    # budget, short of the memory free now with its table: the table reads
    # from the file all the same, and the load prices it so.
    import gmlx.device as device
    import gmlx.load.discovery as discovery
    import gmlx.serve.capacity as cap
    from helpers import FLASH_NEXT, flash_next_under_its_table

    for k in ("GMLX_OVERCOMMIT", "GMLX_WIRE_CHUNK_GB", "GMLX_STREAM_PLE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(budget, "_WIRE_DEPTH", None)
    monkeypatch.setattr(budget, "WIRE_EXEMPT_BYTES", 0)
    monkeypatch.setattr(
        budget, "kv_room_bytes",
        lambda p, env=None, step=None, depth=None, resident=False: budget.KvRoom(
            0, 32768, 1, 0, 0, 0, priced=True))
    table = flash_next_under_its_table(monkeypatch, tmp_path)
    path = str(tmp_path / FLASH_NEXT)
    _box(host_box.mac(128)[0], unwired=90 * GiB if fits else GiB // 1024)
    if whole:
        text = discovery.text_tensor_bytes(path)
        monkeypatch.setattr(discovery, "_stream_budget_bytes", lambda: text)
        monkeypatch.setattr(device, "stream_budget_bytes", lambda: text)
        assert cap.deferred_table_bytes(path) == 0
        _box(host_box.mac(128)[0], unwired=budget.kernel_floor_bytes() + text - table // 2)
    args = _args(path, no_speculative=True)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is not fits
    err = capsys.readouterr().err
    # Each load names the table it reads from the file.
    line = ("[stream] per-layer embedding table of "
            f"{table / 1e9:.1f} GB offloaded to disk\n")
    if fits:
        assert err == line
        assert cap.deferred_table_bytes(path) == table
    else:
        assert err.startswith(line + f"[stream] {budget.STREAM_FREE_NOW}: ")
        assert err.rstrip().endswith(budget.OVERCOMMIT_LEVER)


def test_no_mtp_leaves_the_head_out(glm_head):
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    args = _args(glm_head, no_speculative=True)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False


def test_the_companion_files_count_in_the_wire(header, tmp_path):
    path = header(70 * GiB, experts=60 * GiB, budget_bytes=200 * GiB,
                  room=0)
    # Kept: the 4 GB kernel floor.
    _box(host_box.mac(128)[0], unwired=75 * GiB)
    args = _args(path)
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is False      # 70 <= 75 - 3.7
    big = tmp_path / "mmproj.gguf"
    with open(big, "wb") as f:
        f.truncate(2 * GiB)
    args = _args(path, mmproj=str(big))
    cli.preload_wire_check(args, depth=None)
    assert args.stream_experts is True       # 72 > 71.3


# --- the run's room depth ---------------------------------------------------


@pytest.fixture(autouse=True)
def _run_gguf(tmp_path, monkeypatch):
    """The model file a run's depth is priced for."""
    f = tmp_path / "run.gguf"
    f.write_bytes(b"x")
    monkeypatch.setattr(sys.modules[__name__], "_RUN_GGUF", str(f))


_RUN_GGUF = ""


def _run_args(**kw):
    base = dict(gguf=_RUN_GGUF, speculative=None, draft_gguf=None,
                bench="", bench_depths="", bench_decode_tokens=None,
                max_tokens=None, prompt="Hello, world!", prompt_file="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_capped_run_prices_its_prompt_and_reply():
    assert cli.run_room_depth(_run_args(
        prompt="x" * 1000, max_tokens=256)) == 500 + cli._TEMPLATE_TOKENS + 256


def test_the_system_prompt_counts_and_a_raw_prompt_has_no_template():
    assert cli.run_room_depth(_run_args(
        prompt="x" * 1000, system_prompt="s" * 600,
        max_tokens=256)) == 800 + cli._TEMPLATE_TOKENS + 256
    assert cli.run_room_depth(_run_args(
        prompt="x" * 1000, no_chat_template=True,
        max_tokens=256)) == 500 + 256


def test_a_prompt_file_prices_its_size(tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("y" * 4000)
    assert cli.run_room_depth(_run_args(
        prompt_file=str(pf), max_tokens=100)) == 2100 + cli._TEMPLATE_TOKENS


@pytest.mark.parametrize("kw", [dict(max_tokens=None),
                                dict(max_tokens=0),
                                dict(max_tokens=64, speculative=True),
                                dict(max_tokens=64, draft_gguf="d.gguf"),
                                dict(max_tokens=64, prompt_file="/no/such"),
                                dict(max_tokens=64, image="cat.png"),
                                dict(max_tokens=64, audio="a.wav")])
def test_these_runs_price_the_serve_context(kw):
    assert cli.run_room_depth(_run_args(**kw)) is None


def test_a_run_that_loads_the_model_s_own_mtp_head_prices_the_serve_context(
        monkeypatch):
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: True)
    monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
    args = _run_args(max_tokens=64)
    assert cli.run_room_depth(args) is None
    assert cli.run_room_depth(args, mtp=False) == 6 + cli._TEMPLATE_TOKENS + 64
    args.no_speculative = True
    assert cli.run_room_depth(args) == 6 + cli._TEMPLATE_TOKENS + 64


def test_a_run_of_no_file_prices_no_depth():
    assert cli.run_room_depth(_run_args(gguf="/no/such.gguf",
                                        max_tokens=64)) is None


def test_a_missing_file_with_native_mtp_is_named(tmp_path, capsys):
    # The depth resolves no MTP for a path that is no file, so the run's
    # own check names it.
    missing = str(tmp_path / "missing.gguf")
    assert cli.main([missing, "--native-mtp", "--prompt", "hi"]) == 2
    assert capsys.readouterr().err.startswith(
        f"error: no such file: {missing}")


def test_a_bench_prices_the_deepest_row_of_the_bench_that_runs():
    # --bench decodes 32 tokens a row, --bench-depths 128, and with both
    # given --bench-depths runs.
    assert cli.run_room_depth(_run_args(bench="512,16384")) == 16384 + 32
    assert cli.run_room_depth(_run_args(bench_depths="0,32768",
                                        bench_decode_tokens=64)) == 32768 + 64
    assert cli.run_room_depth(_run_args(bench="65536",
                                        bench_depths="0,8192")) == 8192 + 128
    assert cli.run_room_depth(_run_args(bench="a,b")) is None


# --- where the check runs -----------------------------------------------------


@pytest.mark.parametrize("head", [False, True])
def test_run_checks_before_the_stream_latch(monkeypatch, head):
    # With the model's own MTP head auto-loading, the check prices the
    # serve context, and the run's own room for a set without the head.
    seen = []
    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a: None)
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda p: head)
    monkeypatch.setattr(cli, "_sibling_drafter", lambda p: None)
    monkeypatch.setattr(cli, "preload_wire_check",
                        lambda args, depth, **k: seen.append(
                            ("check", depth, k["depth_without_mtp"])))
    monkeypatch.setattr(cli, "_ensure_stream_cb_caps",
                        lambda args: seen.append(("latch",)))
    monkeypatch.setattr(cli, "_run_generate", lambda args: 0)
    run = 1 + cli._TEMPLATE_TOKENS + 8
    assert cli.main([_RUN_GGUF, "--prompt", "hi", "--max-tokens", "8"]) == 0
    assert seen == [("check", None if head else run, run), ("latch",)]


def test_a_run_s_depth_does_not_outlive_it(monkeypatch):
    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a: None)
    monkeypatch.setattr(cli, "_ensure_stream_cb_caps", lambda args: None)
    monkeypatch.setattr(cli, "_run_generate", lambda args: 0)
    monkeypatch.setattr(budget, "_WIRE_DEPTH", None)
    seen = []
    real = cli.preload_wire_check

    def check(args, depth, **k):
        real(args, depth=depth, **k)
        seen.append(budget.wire_room_depth())
    monkeypatch.setattr(cli, "preload_wire_check", check)
    assert cli.main([_RUN_GGUF, "--prompt", "hi", "--max-tokens", "8"]) == 0
    assert seen == [1 + cli._TEMPLATE_TOKENS + 8]
    assert budget.wire_room_depth() is None


def test_chat_checks_at_the_serve_context_before_the_stream_latch(
        monkeypatch, tmp_path):
    from gmlx.tui import chat

    seen = []
    monkeypatch.setattr(cli, "maybe_load_from_config", lambda *a: None)
    monkeypatch.setattr(cli, "preload_wire_check",
                        lambda args, depth, **k: seen.append(("check", depth)))
    monkeypatch.setattr(cli, "_ensure_stream_cb_caps",
                        lambda args: seen.append(("latch", None)))
    missing = str(tmp_path / "missing.gguf")
    assert chat.cmd_chat([missing]) == 2
    assert seen == [("check", None), ("latch", None)]
