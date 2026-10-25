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

    for k in ("GMLX_STREAM_KV_CTX", "GMLX_STREAM_KV_WIDTH", "GMLX_DECODE_BATCH",
              "GMLX_ADMIT_RESERVE_GB",
              "GMLX_DECODE_KV_RESERVE_GB", "GMLX_GOV_MARGIN",
              "GMLX_STREAM_CACHE_GB", "PREFILL_STEP_SIZE",
              "GMLX_STREAM_PREFILL_TAIL_MERGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG))
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 100.0 * GB)
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


QWEN4EXP_CFG = {"model_type": "qwen4_exp", "hidden_size": 2560,
                "num_attention_heads": 24, "head_dim": 256, "hc_count": 4,
                "num_experts_per_tok": 10, "moe_intermediate_size": 640}


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


@pytest.mark.parametrize("cfg,rows", [
    # Qwen3.8-Flash-Next: its arch prefills 8192 rows, with no merged tail.
    (dict(QWEN4EXP_CFG, num_experts=512), 8192),
    # DeepSeek V4.1 Flash: its arch's 4096.
    (dict(DS41_CFG, n_routed_experts=256), 4096),
    # Qwen3.6-35B-A3B, 256 experts at 8 a token: the wide MoE step.
    (dict(model_type="qwen3_5_moe", hidden_size=2048,
          num_attention_heads=16, head_dim=256, num_experts=256,
          num_experts_per_tok=8, moe_intermediate_size=512), 4096),
    # gpt-oss-20b, 32 experts at 4 a token: the stock 2048.
    (dict(model_type="gpt_oss", hidden_size=2880, num_attention_heads=64,
          head_dim=64, num_local_experts=32, num_experts_per_tok=4,
          intermediate_size=2880), 2048),
])
def test_a_moe_kept_in_memory_prices_the_chunk_it_prefills(
        chunk_env, monkeypatch, cfg, rows):
    """A load that keeps the experts in memory prefills at the step its
    arch or its expert count takes, so its keep prices those rows, not
    the streamed chunk's. A requested step wins."""
    for k in ("GMLX_MOE_WIDE_STEP", *budget._ARCH_STEP_ENV.values()):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GMLX_STREAM_PREFILL_TAIL_MERGE", "0")
    mt = cfg["model_type"]
    per_row = budget.prefill_activation_bytes(cfg) // budget.prefill_rows(mt)
    assert budget.prefill_activation_bytes(cfg, resident=True) == per_row * rows
    assert budget.prefill_activation_bytes(
        cfg, 1024, resident=True) == per_row * 1024


def test_the_resident_arch_steps_are_the_archs_own(monkeypatch):
    import gmlx.models.deepseek_v4.model as v4
    import gmlx.models.deepseek_v41.model as v41
    import gmlx.models.qwen4_exp.model as q4

    assert budget._ARCH_RESIDENT_STEP == {
        "qwen4_exp": q4._Q4_BASE_STEP, "deepseek_v4": v4._V4_BASE_STEP,
        "deepseek_v41": v41._DS41_BASE_STEP}
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    monkeypatch.setenv("GMLX_Q4_PREFILL_STEP", "2048")
    assert budget.resident_prefill_rows(None, "qwen4_exp") == 2048
    # A batch the arch's profile is off for takes the wide MoE step.
    monkeypatch.setenv("GMLX_MOE_WIDE_STEP", "16384")
    assert budget.resident_prefill_rows(
        None, "deepseek_v41", experts=256, used=6) == 16384


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
    assert budget.prefill_activation_bytes(QWEN4EXP_CFG) == 4_331_667_456
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


# Qwen3.8-Flash-Next's header widths: 4 residual streams.
def test_room_holds_the_chunk_and_a_2_gib_pool(header, monkeypatch):
    """The room holds one chunk's activations (8.0 GB on DeepSeek V4.1
    Flash and 4.3 GB on Qwen3.8-Flash-Next at the streamed chunk width)
    and a 2 GiB pool beside them, whatever the chunk: a pool the size of
    the chunk costs the decode arena more than it gives prefill. Under
    prefill the pool fills to its limit and the live chunk sits on top,
    so the room holds both."""
    import gmlx.commands.tool_preflight as tp

    rooms = {}
    for name, widths in (("ds41", DS41_CFG), ("qwen4exp", QWEN4EXP_CFG)):
        monkeypatch.setattr(tp, "_synth_config",
                            lambda p, w=widths: dict(CFG, **w))
        rooms[name] = budget.kv_room_bytes(header)
        act = budget.prefill_activation_bytes(widths)
        assert rooms[name].activation_bytes == act
        assert rooms[name].pool_bytes == 2 << 30
    assert rooms["ds41"].activation_bytes == 8_021_606_400
    assert rooms["qwen4exp"].activation_bytes == 4_331_667_456
    ds41 = rooms["ds41"]
    wide = dict(DS41_CFG, num_experts_per_tok=16)
    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG, **wide))
    room = budget.kv_room_bytes(header)
    assert room.activation_bytes > ds41.activation_bytes
    assert room.pool_bytes == 2 << 30
    assert room.bytes - ds41.bytes == (
        room.activation_bytes - ds41.activation_bytes)
    # A chunk under 2 GiB keeps the same pool.
    monkeypatch.setenv("PREFILL_STEP_SIZE", "1024")
    small_chunk = budget.kv_room_bytes(header)
    assert small_chunk.activation_bytes < 2 << 30
    assert small_chunk.pool_bytes == 2 << 30
    monkeypatch.delenv("PREFILL_STEP_SIZE")
    # GMLX_STREAM_CACHE_GB sets the pool, and the room follows it.
    monkeypatch.setenv("GMLX_STREAM_CACHE_GB", "1")
    small = budget.kv_room_bytes(header)
    assert small.pool_bytes == 1 << 30
    assert room.bytes - small.bytes == room.pool_bytes - (1 << 30)
    # The fit leaves out the chunk and its pool.
    assert small.fit_bytes == room.fit_bytes


def test_room_reserve_covers_a_batch_at_the_decode_cap(header, monkeypatch):
    """Serve decodes up to the decode batch cap on a streamed model, so the
    room's admission reserve prices that many rows at the room's depth."""
    monkeypatch.delenv("GMLX_ADMIT_RESERVE_GB", raising=False)
    monkeypatch.setenv("GMLX_DECODE_BATCH", "8")
    assert budget.kv_room_bytes(header).reserve_bytes == int(1e9)
    monkeypatch.setenv("GMLX_DECODE_BATCH", "64")
    wide = budget.kv_room_bytes(header)
    layer = BPT // 10 * 32768
    assert wide.reserve_bytes == 64 * layer + layer
    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    assert budget.kv_room_bytes(header).reserve_bytes == int(2e9)


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


# Mach's free and file-backed counts both hold the 4 GiB of speculative
# pages: 6 GiB free, 1 GiB purgeable and 20 GiB of page cache.
_MACH = {"free": 10 << 30, "inactive": 0, "purgeable": 1 << 30,
         "speculative": 4 << 30, "filebacked": 20 << 30, "wired": 0,
         "compressor": 0}


def test_the_kernel_reclaimable_sum_counts_the_speculative_pages_once(
        monkeypatch):
    import gmlx.serve.kernel_vm as kv

    monkeypatch.setattr(kv, "snapshot", lambda: dict(_MACH))
    assert kv.reclaimable_bytes() == 27 << 30
    assert budget.reclaimable_ram_bytes() == 27 << 30


def test_the_arena_sizing_read_is_live(monkeypatch):
    # The mach snapshot can be a second old, and a wire of tens of GB
    # commits within one refresh.
    import gmlx.serve.kernel_vm as kv

    monkeypatch.setattr(kv, "snapshot", lambda: dict(_MACH))
    monkeypatch.setattr(kv, "live_reclaimable_bytes", lambda: 9 << 30)
    assert budget._available_ram_bytes() == 9 << 30
    # The strict set leaves the page cache out, so it never reads live.
    assert budget._available_ram_bytes(include_inactive=False) == 11 << 30
    monkeypatch.setattr(kv, "live_reclaimable_bytes", lambda: None)
    assert budget._available_ram_bytes() == 27 << 30


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
    monkeypatch.setattr(budget, "unwired_ram_bytes", lambda *a, **k: None)
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 << 30)
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": 128 << 30})
    experts = 20 << 30
    kw = dict(total_bytes=30 << 30, offsets={0: [(0, 0, experts)]},
              budget=100 << 30, room_bytes=4 << 30)
    assert budget._decode_arena_bytes(**kw) == experts
    assert budget._decode_arena_bytes(**kw, forced=True) == experts // 2


def test_the_arena_takes_other_apps_memory_but_not_another_models(
        monkeypatch):
    """Apps hold most of the RAM beside a streamed model: free and the page
    cache read near empty, but the kernel compresses their memory, so the
    arena keeps its full size. Memory another model holds wired shrinks
    it."""
    import mlx.core as mx

    import gmlx.serve.kernel_vm as kv
    import gmlx.stream.budget as budget

    for k in ("GMLX_DECODE_ARENA_GB", "GMLX_DECODE_ARENA_RAM_FRAC",
              "GMLX_DECODE_RAM_FLOOR_GB", "GMLX_DECODE_PAGECACHE_GB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 << 30)
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": 128 << 30})
    monkeypatch.setattr(kv, "reclaimable_bytes", lambda: 3 << 30)
    monkeypatch.setattr(kv, "live_reclaimable_bytes", lambda: 3 << 30)
    experts = 20 << 30
    kw = dict(total_bytes=30 << 30, offsets={0: [(0, 0, experts)]},
              budget=100 << 30, room_bytes=4 << 30)
    monkeypatch.setattr(kv, "unwired_bytes", lambda: 110 << 30)
    assert budget._decode_arena_bytes(**kw) == experts
    monkeypatch.setattr(kv, "unwired_bytes", lambda: 30 << 30)
    assert 0 < budget._decode_arena_bytes(**kw) < experts


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
    monkeypatch.setattr(mx, "device_info",
                        lambda *a: {"memory_size": 1000 << 30})
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 << 30)
    avail = {"v": None, "easy": None}
    monkeypatch.setattr(budget, "unwired_ram_bytes",
                        lambda *a, **k: avail["v"])
    monkeypatch.setattr(budget, "_available_ram_bytes",
                        lambda *a, **k: avail["easy"] or avail["v"])
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
    # With 30 GiB of it other apps' memory, those count at half: the
    # kernel keeps what it compresses in RAM.
    avail["easy"] = 30 * gib
    assert budget._decode_arena_bytes(**kw) == (47 + 17) // 2 * gib
    avail["easy"] = None
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
    ceiling leaves 39.5 GB of RAM outside it, which holds the floor and
    the host overhead, so the reclaimable leg binds there too. An arena
    under 50 GB is the arena that ran decode 15 to 20% slower, for free
    memory nothing needs."""
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
    monkeypatch.setattr(budget, "unwired_ram_bytes", lambda *a, **k: avail)
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


# Kimi-K3 UD-Q2_K_XL boxes where the ceiling leg binds: stock 128 GB and
# 512 GB, where the RAM outside the ceiling holds the floor and the host
# overhead, and 128 GB with a raised GPU limit, where 1.6 GB of them are
# past it.
@pytest.mark.parametrize("ram_gib,ws,avail", [
    (128, 0.75 * (128 << 30), 120e9),
    (128, 125e9, 120e9),
    (512, 0.75 * (512 << 30), 530e9),
], ids=["128-stock", "128-raised", "512-stock"])
def test_loader_arena_matches_the_plan_where_the_ceiling_binds(
        monkeypatch, ram_gib, ws, avail):
    """The loader sizes the arena the plan prints. Both charge the floor
    and the host overhead under the ceiling only past the RAM the ceiling
    leaves outside it. On a stock 128 GB Mac the loader's arena was 9.6 GB
    under the plan's 14.3 GB, and the ring the plan builds on 192 GB and
    up was judged against an arena short by the whole floor."""
    import mlx.core as mx

    import gmlx.stream.plan as sp

    for k in ("GMLX_DECODE_ARENA_GB", "GMLX_DECODE_ARENA_FORCE",
              "GMLX_DECODE_ARENA_RAM_FRAC", "GMLX_DECODE_RAM_FLOOR_GB",
              "GMLX_DECODE_PAGECACHE_GB", "GMLX_DECODE_PROCESS_GB",
              "GMLX_DECODE_RESIDENT_CACHE_GB", "GMLX_STREAM_CACHE_GB",
              "GMLX_GOV_MARGIN", "GMLX_GOV_RESERVE_GB", "GMLX_RING_TIMESHARE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "20")
    ram = ram_gib << 30
    every, experts, ring = 62.2e9, 799.1e9, 22.7e9
    model = sp.ModelPlan(
        arch="deepseek2", total_bytes=int(every + experts),
        expert_bytes=int(experts), groups={}, moe_layers=92, n_experts=896,
        experts_per_token=16, ring_bytes=int(ring), kv_costs=None,
        trained_ctx=None)
    box = sp.box_plan(model, ram_bytes=ram, ws_bytes=ws)
    monkeypatch.setattr(mx, "device_info", lambda *a: {"memory_size": ram})
    monkeypatch.setattr(budget, "ceiling_bytes",
                        lambda: sp.ceiling_for(ram, ws))
    monkeypatch.setattr(budget, "unwired_ram_bytes", lambda *a, **k: avail)
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda *a, **k: avail)

    def loader(ring_bytes=0):
        return budget._decode_arena_bytes(
            model.total_bytes, {0: [(0, 0, model.expert_bytes)]},
            int(sp.ceiling_for(ram, ws)), room_bytes=box.room.bytes,
            pinned_bytes=int(every), ring_bytes=ring_bytes,
            host_bytes=budget.host_overhead_bytes(ram))

    arena = loader()
    if box.ring_fits:
        held = loader(int(ring))
        if not budget.ring_share_bytes(arena, held, int(ring)):
            arena = held
    assert arena == pytest.approx(box.arena_bytes, abs=64)
    if ram_gib == 128 and ws < 100e9:
        assert arena == pytest.approx(14.26e9, rel=0.01)
        assert not box.ring_fits


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
