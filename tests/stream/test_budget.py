"""The streaming memory budget: the KV room the arena leaves under the
governor ceiling, and the governor-shifted headroom the regrow gate reads."""

from types import SimpleNamespace

import pytest

import gmlx.serve.capacity as cap
import gmlx.stream.budget as budget

GB = 1e9

CFG = {
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 65536,
}
BPT = 2 * 8 * 64 * 2 * 10  # 20480 B/token across layers


@pytest.fixture
def header(monkeypatch, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    import gmlx.commands.tool_preflight as tp
    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.memory as sm

    for k in ("GMLX_STREAM_KV_CTX", "GMLX_STREAM_KV_WIDTH",
              "GMLX_DECODE_KV_RESERVE_GB", "GMLX_GOV_MARGIN",
              "GMLX_STREAM_CACHE_GB", "PREFILL_STEP_SIZE",
              "GMLX_STREAM_PREFILL_TAIL_MERGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG))
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 100.0 * GB)
    monkeypatch.setattr(sm, "admit_reserve_bytes", lambda ws, gen=None: 1.0 * GB)
    monkeypatch.setattr(pd, "_cap_bytes", lambda: 3.0 * GB)
    return str(f)


def test_room_prices_ctx_width_transient_reserve(header, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_KV_CTX", "4096")
    monkeypatch.setenv("GMLX_STREAM_KV_WIDTH", "2")
    room = budget.kv_room_bytes(header)
    assert room.priced
    assert (room.depth, room.width) == (4096, 2)
    assert room.kv_bytes == BPT * 4096 * 2
    assert room.transient_bytes == int(3.0 * GB)
    assert room.reserve_bytes == int(1.0 * GB)
    # A config without the chunk widths prices no activations and keeps
    # the allocator's 4 GiB pool, which the room still holds.
    assert room.activation_bytes == 0 and room.pool_bytes == 4 << 30
    assert room.bytes == (room.kv_bytes + room.transient_bytes
                          + room.pool_bytes + room.reserve_bytes)


# The deepseek41 header's synthesized widths.
DS41_CFG = {"model_type": "deepseek_v41", "hidden_size": 5120,
            "num_attention_heads": 64, "head_dim": 512, "hc_mult": 4,
            "num_experts_per_tok": 6, "moe_intermediate_size": 2304}


@pytest.fixture
def chunk_env(monkeypatch):
    for k in ("PREFILL_STEP_SIZE", "GMLX_STREAM_PREFILL_TAIL_MERGE"):
        monkeypatch.delenv(k, raising=False)


def test_chunk_rows_follow_the_streamed_chunk_width(chunk_env, monkeypatch):
    for k in budget._ARCH_STEP_ENV.values():
        monkeypatch.delenv(k, raising=False)
    # The loader's 8192-row streaming step, 9/8 of it with the tail merge.
    assert budget.prefill_rows("deepseek_v41") == 9216
    assert budget.prefill_rows("hy_v4") == 4608
    # The arch's own step env, as its streamed prefill takes it.
    monkeypatch.setenv("GMLX_DS41_PREFILL_STEP", "16384")
    monkeypatch.setenv("GMLX_Q4_PREFILL_STEP", "0")
    assert budget.prefill_rows("deepseek_v41") == 18432
    assert budget.prefill_rows("qwen4_exp") == 9216
    assert budget.prefill_rows("hy_v4") == 4608
    # An explicit step wins: the server's PREFILL_STEP_SIZE, and the
    # step run and chat ask for.
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert budget.prefill_rows("deepseek_v41") == 2304
    assert budget.prefill_rows("deepseek_v41", 4096) == 4608
    monkeypatch.setenv("GMLX_STREAM_PREFILL_TAIL_MERGE", "0")
    assert budget.prefill_rows("deepseek_v41") == 2048


def test_chunk_activations_cover_the_measured_deepseek41_prefill(
        chunk_env, monkeypatch):
    """DeepSeek V4.1 Flash UD-IQ2_XXS held 5.4 GiB of live activations
    over the buffer pool under an 8034-token prefill on an M5 Max. The
    fp32 price covers it without doubling it."""
    monkeypatch.setenv("PREFILL_STEP_SIZE", "8034")
    monkeypatch.setenv("GMLX_STREAM_PREFILL_TAIL_MERGE", "0")
    act = budget.prefill_activation_bytes(DS41_CFG)
    assert 5.4 * (1 << 30) <= act <= 1.3 * 5.4 * (1 << 30)


def test_chunk_activations_read_each_archs_width_names(chunk_env):
    """The synth names the widths per arch: qwen4exp's residual streams
    are hc_count, ernie4_5-moe's top-k is moe_k, hunyuan's moe_topk,
    gemma4's top_k_experts, hy_v3's expert width expert_hidden_dim, and
    nemotron_h puzzle models list k and the expert width per layer. Each
    prices what the same widths do under the common names."""
    # Qwen3.8-Flash-Next's header: 4 residual streams.
    qwen = dict(model_type="qwen4_exp", hidden_size=2560,
                num_attention_heads=24, head_dim=256, hc_count=4,
                num_experts_per_tok=10, moe_intermediate_size=640)
    assert budget.prefill_activation_bytes(qwen) == 4_331_667_456
    # ERNIE-4.5-300B-A47B's widths, top-k under moe_k.
    ernie = dict(model_type="ernie4_5_moe", hidden_size=8192,
                 num_attention_heads=64, head_dim=128, moe_k=8,
                 moe_intermediate_size=3584)
    assert budget.prefill_activation_bytes(ernie) == 9_512_681_472
    common = dict(hidden_size=4096, num_attention_heads=32, head_dim=128,
                  num_experts_per_tok=8, moe_intermediate_size=1536)
    want = budget.prefill_activation_bytes(common)
    for key in ("moe_topk", "top_k_experts"):
        alias = dict(common)
        alias[key] = alias.pop("num_experts_per_tok")
        assert budget.prefill_activation_bytes(alias) == want
    hy = dict(common, intermediate_size=12288)
    hy["expert_hidden_dim"] = hy.pop("moe_intermediate_size")
    assert budget.prefill_activation_bytes(hy) == want
    # A per-layer list prices its widest layer; 0 marks a dense layer.
    nemo = dict(common, num_experts_per_tok=[0, 6, 8],
                moe_intermediate_size=[0, 1024, 1536])
    assert budget.prefill_activation_bytes(nemo) == want
    # Experts behind a latent projection take their input at its width.
    latent = dict(common, routed_expert_hidden_size=1024)
    assert budget.prefill_activation_bytes(latent) < want
    # A nested text config, and configs without the widths.
    assert budget.prefill_activation_bytes({"text_config": common}) == want
    assert budget.prefill_activation_bytes({}) == 0
    assert budget.prefill_activation_bytes({"hidden_size": 1024}) == 0


def test_room_holds_the_chunk_and_the_pool_the_allocator_keeps(
        header, monkeypatch):
    """The pool is a quarter of the chunk, at least 2 GiB: on DeepSeek V4.1
    Flash's 8 GB chunk a 2 GiB pool prefilled 2k and 8k prompts as fast as
    the whole chunk did. Under prefill it fills to its limit and the live
    chunk sits on top, so the room moves with both."""
    import gmlx.commands.tool_preflight as tp

    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG, **DS41_CFG))
    ds41 = budget.kv_room_bytes(header)
    assert ds41.activation_bytes == budget.prefill_activation_bytes(DS41_CFG)
    assert ds41.pool_bytes == 2 << 30
    wide = dict(DS41_CFG, num_experts_per_tok=16)
    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG, **wide))
    room = budget.kv_room_bytes(header)
    assert room.pool_bytes == room.activation_bytes // 4 > 2 << 30
    assert room.bytes - ds41.bytes == (
        room.activation_bytes - ds41.activation_bytes
        + room.pool_bytes - ds41.pool_bytes)
    # GMLX_STREAM_CACHE_GB sets the pool, and the room follows it.
    monkeypatch.setenv("GMLX_STREAM_CACHE_GB", "1")
    small = budget.kv_room_bytes(header)
    assert small.pool_bytes == 1 << 30
    assert room.bytes - small.bytes == room.pool_bytes - (1 << 30)
    # The fit leaves out the chunk and its pool.
    assert small.fit_bytes == room.fit_bytes


def test_room_depth_clamped_to_trained_context(header, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_KV_CTX", str(1 << 20))
    room = budget.kv_room_bytes(header)
    assert room.depth == 65536
    assert room.kv_bytes == BPT * 65536


def test_room_default_depth(header):
    room = budget.kv_room_bytes(header)
    assert room.depth == 32768 and room.width == 1


def test_flat_room_on_env_override(header, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "3")
    room = budget.kv_room_bytes(header)
    assert not room.priced
    assert room.bytes == 3 << 30


def test_flat_room_when_header_unpriceable(header, monkeypatch):
    import gmlx.commands.tool_preflight as tp

    def _boom(p):
        raise OSError("no header")

    monkeypatch.setattr(tp, "_synth_config", _boom)
    room = budget.kv_room_bytes(header)
    assert not room.priced
    assert room.bytes == 8 << 30
    assert budget.kv_room_bytes(None).bytes == 8 << 30


def test_ceiling_and_governor_headroom(monkeypatch):
    import mlx.core as mx
    import gmlx.gen.prefill_decay as pd

    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 100.0 * GB)
    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 128.0 * GB})
    # 0.95 x ws binds below RAM minus the 10% reserve (115.2 GB).
    assert budget.ceiling_bytes() == pytest.approx(95.0 * GB)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 30.0 * GB)
    assert budget.governor_headroom_bytes() == pytest.approx(25.0 * GB)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: None)
    assert budget.ceiling_bytes() is None
    assert budget.governor_headroom_bytes() is None


def test_kv_room_dataclass_is_frozen():
    room = budget.KvRoom(1, 2, 3, 4, 5, 6, priced=True)
    with pytest.raises(Exception):
        room.bytes = 7
    assert isinstance(SimpleNamespace(**room.__dict__).bytes, int)


def test_legacy_room_ignores_a_malformed_reserve(monkeypatch):
    from gmlx.stream.budget import legacy_room_bytes

    for bad in ("abc", "1e999", "nan"):
        monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", bad)
        assert legacy_room_bytes() == 8 << 30
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "2")
    assert legacy_room_bytes() == 2 << 30


def test_transient_bytes_follows_the_decay_cap(monkeypatch):
    from gmlx.stream.budget import transient_bytes

    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB", raising=False)
    assert transient_bytes(100e9) == 5e9
    assert transient_bytes(10e9) == 2e9
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "3")
    assert transient_bytes(100e9) == 3e9


def test_reclaimable_ram_bytes_prefers_the_kernel_counters(monkeypatch):
    import gmlx.serve.kernel_vm as kv

    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: 12.5e9)
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda include_inactive=True: 3)
    assert budget.reclaimable_ram_bytes() == int(12.5e9)
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: None)
    assert budget.reclaimable_ram_bytes() == 3


def test_kernel_floor_bytes_is_the_governor_floor(monkeypatch):
    monkeypatch.setenv("GMLX_GOV_KERNEL_FLOOR_GB", "2")
    assert budget.kernel_floor_bytes() == 2e9


def test_available_ram_reads_the_mach_counters_without_a_spawn(monkeypatch):
    """The loader's reclaimable snapshot comes from host_statistics64; the
    vm_stat spawn is the fallback only. A fork beside a Metal-mapped
    arena copies the arena, so the serve process must not fork here."""
    import subprocess

    from gmlx.stream.budget import _available_ram_bytes
    from gmlx.serve import kernel_vm

    if kernel_vm.snapshot() is None:
        pytest.skip("mach counters unavailable")

    def boom(*a, **k):
        raise AssertionError("spawned")

    monkeypatch.setattr(subprocess, "run", boom)
    loose, strict = _available_ram_bytes(), _available_ram_bytes(include_inactive=False)
    assert loose > 0 and strict > 0 and loose >= strict


def test_a_forced_stream_under_the_budget_wires_half_the_experts(monkeypatch):
    """--stream-experts on a model that fits: an arena of every expert would
    wire them all, which is the resident load the flag asks to avoid."""
    import mlx.core as mx

    import gmlx.stream.budget as budget

    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_ARENA_RAM_FRAC", raising=False)
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 << 30)
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": 128 << 30})
    experts = 20 << 30
    kw = dict(total_bytes=30 << 30, offsets={0: [(0, 0, experts)]},
              budget=100 << 30, room_bytes=4 << 30)
    assert budget._decode_arena_bytes(**kw) == experts
    assert budget._decode_arena_bytes(**kw, forced=True) == experts // 2


def test_host_overhead_is_priced_at_its_cap(monkeypatch):
    """DeepSeek V4.1 Flash served on a 128 GB M5 Max held 1.8 GiB of
    process memory and 8.6 to 8.7 GiB of file pages outside MLX under
    prefill: 10.4 to 10.5 GiB against the 10 GiB priced. The price does
    not read the page cache at sizing, so a cold load sizes the arena a
    warm one does."""
    for k in ("GMLX_DECODE_PROCESS_GB", "GMLX_DECODE_RESIDENT_CACHE_GB"):
        monkeypatch.delenv(k, raising=False)
    gib = 1 << 30
    assert budget.host_overhead_bytes(128 * gib) == 10 * gib
    # 1/16 of RAM, at most 8 GiB, on top of 2 GiB of process memory.
    assert budget.host_overhead_bytes(64 * gib) == 6 * gib
    assert budget.host_overhead_bytes(512 * gib) == 10 * gib
    monkeypatch.setenv("GMLX_DECODE_PROCESS_GB", "0")
    monkeypatch.setenv("GMLX_DECODE_RESIDENT_CACHE_GB", "3")
    assert budget.host_overhead_bytes(128 * gib) == 3 * gib


def test_arena_charges_the_host_overhead_on_ram_not_the_ceiling(monkeypatch):
    """The host overhead is not MLX memory: the RAM the ceiling leaves
    outside it holds that, so only the reclaimable leg and an explicit
    arena's clamp, which measure RAM, charge it."""
    import mlx.core as mx

    for k in ("GMLX_DECODE_ARENA_GB", "GMLX_DECODE_ARENA_FORCE",
              "GMLX_DECODE_ARENA_RAM_FRAC"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GMLX_DECODE_RAM_FLOOR_GB", "0")
    monkeypatch.setenv("GMLX_DECODE_PAGECACHE_GB", "0")
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": 1000 << 30})
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 << 30)
    avail = {"v": None}
    monkeypatch.setattr(budget, "_available_ram_bytes",
                        lambda *a, **k: avail["v"])
    gib = 1 << 30
    kw = dict(total_bytes=210 * gib, offsets={0: [(0, 0, 200 * gib)]},
              budget=100 * gib, room_bytes=10 * gib, pinned_bytes=10 * gib)
    # The ceiling leg: 100 less 10 every-token and 10 room, whatever the
    # host overhead.
    assert budget._decode_arena_bytes(**kw, host_bytes=3 * gib) == 80 * gib
    assert budget._decode_arena_bytes(**kw, host_bytes=0) == 80 * gib
    # The reclaimable leg: the pin is already out of the 60 GiB snapshot.
    avail["v"] = 60 * gib
    kw["host_bytes"] = 3 * gib
    assert budget._decode_arena_bytes(**kw) == 47 * gib
    # An explicit arena clamps to the same: the next prefill's transients
    # and the resident pages need the room it would take.
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "50")
    assert budget._decode_arena_bytes(**kw) == 47 * gib
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "40")
    assert budget._decode_arena_bytes(**kw) == 40 * gib
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "50")
    monkeypatch.setenv("GMLX_DECODE_ARENA_FORCE", "1")
    assert budget._decode_arena_bytes(**kw) == 50 * gib


# Working sets of a 128 GB M5 Max: the raised GPU wired limit the
# measurements ran at, and the stock limit, 3/4 of RAM.
@pytest.mark.parametrize("ws", [120.2e9, 0.75 * (128 << 30)],
                         ids=["raised", "stock"])
def test_deepseek41_default_arena_keeps_8_gib_free_at_the_prefill_peak(
        monkeypatch, ws):
    """DeepSeek V4.1 Flash UD-IQ2_XXS served by default on a 128 GB M5
    Max. At load: 102.0 GB reclaimable, 8.4 GB of every-token weights
    (pinned), a 135.2 GB table and 156.9 GB of experts on disk, a 9.27 GB
    ring the arena lends at prefill. Measured under an 8k prefill: the
    buffer pool at its limit, 5.4 GiB of live chunk activations on top,
    8.7 GiB of file pages and 1.8 GiB of process memory outside MLX that
    stayed resident. The room without the chunk, the pool or the host
    overhead (12.0 GB, and the pool set to all of it) sized an 80.4 GB
    arena that left the box 7 GiB short. At the stock working set the
    ceiling leg binds: an arena under 50 GB there is the arena that ran
    decode 15 to 20% slower, for free memory nothing needs."""
    import mlx.core as mx

    import gmlx.serve.mem_preflight as mp

    for k in ("GMLX_DECODE_ARENA_GB", "GMLX_DECODE_ARENA_RAM_FRAC",
              "GMLX_DECODE_RAM_FLOOR_GB", "GMLX_DECODE_PAGECACHE_GB",
              "GMLX_DECODE_PROCESS_GB", "GMLX_DECODE_RESIDENT_CACHE_GB",
              "GMLX_STREAM_CACHE_GB", "GMLX_STREAM_KV_CTX",
              "GMLX_STREAM_KV_WIDTH", "GMLX_DECODE_KV_RESERVE_GB",
              "GMLX_ADMIT_RESERVE_GB", "GMLX_RING_TIMESHARE",
              "PREFILL_STEP_SIZE", "GMLX_STREAM_PREFILL_TAIL_MERGE"):
        monkeypatch.delenv(k, raising=False)
    gib = 1 << 30
    ram, avail = 128 * gib, int(102.0e9)
    every, table, experts, ring = 8.4e9, 135.2e9, 156.9e9, 9.27e9
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": ram})
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 0.95 * ws)
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda *a, **k: avail)
    monkeypatch.setattr(mp, "prompt_kv_bytes", lambda costs, n: 10e6)
    room = budget.price_room(((1, 1),), 1 << 20, ws, 0.05 * ws,
                             budget.prefill_activation_bytes(DS41_CFG))
    kw = dict(total_bytes=int(every + table + experts),
              offsets={0: [(0, 0, int(experts))]}, budget=int(0.95 * ws),
              room_bytes=room.bytes, pinned_bytes=int(every),
              streamable_bytes=int(table),
              host_bytes=budget.host_overhead_bytes(ram))
    full = budget._decode_arena_bytes(**kw)
    held = budget._decode_arena_bytes(**kw, ring_bytes=int(ring))
    share = budget.ring_share_bytes(full, held, int(ring))
    arena = full if share else held
    assert share == int(ring)
    free = avail - arena - room.pool_bytes - 5.4 * gib - 8.7 * gib - 1.8 * gib
    assert free >= 8 * gib
    assert arena >= 50e9


def test_ring_share_is_the_room_the_ring_adds_to_the_arena(monkeypatch):
    monkeypatch.delenv("GMLX_RING_TIMESHARE", raising=False)
    gb = 1 << 30
    assert budget.ring_share_bytes(60 * gb, 52 * gb, 8 * gb) == 8 * gb
    # An arena capped at the expert bytes gains only part of the room.
    assert budget.ring_share_bytes(60 * gb, 57 * gb, 8 * gb) == 3 * gb
    # Lends free whole layers: a small arena keeps the room apart.
    assert budget.ring_share_bytes(15 * gb, 7 * gb, 8 * gb) == 0
    assert budget.ring_share_bytes(60 * gb, 60 * gb, 0) == 0
    monkeypatch.setenv("GMLX_RING_TIMESHARE", "0")
    assert budget.ring_share_bytes(60 * gb, 52 * gb, 8 * gb) == 0
