"""Depth decay keeps a chunk whole when stock MLX runs it fused."""

from types import SimpleNamespace

import pytest

import gmlx.gen.prefill_decay as pd
from gmlx.cache import kv_cache as kc

_NAX_PROBE = pd._nax_available


def _model(head_dim=256, heads=24, model_type="fused_arch", **extra):
    return SimpleNamespace(config=SimpleNamespace(
        model_type=model_type, num_attention_heads=heads, head_dim=head_dim,
        **extra))


def _caches(depth, *types):
    out = []
    for t in types or (kc.ArraysCache, kc.KVCache):
        c = t(1) if t is kc.ArraysCache else t()
        c.offset = depth
        out.append(c)
    return out


def _batch(model, depth, base=2048, caches=None):
    return SimpleNamespace(prefill_step_size=base, model=model,
                           prompt_cache=caches or _caches(depth))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "5.7")
    monkeypatch.delenv("GMLX_PREFILL_SCORE_PROFILE", raising=False)
    monkeypatch.delenv("GMLX_PREFILL_FUSED_256", raising=False)
    monkeypatch.setattr(pd, "_mlx_version", lambda: (0, 32, 3))
    monkeypatch.setattr(pd, "_nax_available", lambda: True)
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    monkeypatch.setattr(pd, "_LIVE_DECODE_ROWS", 0)
    monkeypatch.setattr(pd, "_CHUNK_COST", None)
    monkeypatch.setitem(pd._SCORE_PROFILES, "fused_arch",
                        pd.fused_attention_profile)


@pytest.mark.parametrize("depth", [0, 59_392, 121_856, 200_000, 1_000_000])
def test_fused_head_dim_256_keeps_the_full_chunk(depth):
    # Qwen3.8-27B geometry: the dense model halves 2048 to 512 by 122k.
    assert pd.decayed_for_batch(_batch(_model(), depth)) == 2048


def test_dense_model_tiers_are_unchanged():
    assert pd.decayed_step(2048, 59_392, 24) == 1024
    assert pd.decayed_step(2048, 121_856, 24) == 512


@pytest.mark.parametrize("dim", [64, 80, 96, 128])
def test_always_fused_head_dims_never_decay(dim):
    assert pd.decayed_for_batch(
        _batch(_model(head_dim=dim, heads=64), 2_000_000)) == 2048


def test_head_dim_from_hidden_size():
    m = SimpleNamespace(config=SimpleNamespace(
        model_type="fused_arch", num_attention_heads=32, hidden_size=4096))
    assert pd.decayed_for_batch(_batch(m, 2_000_000)) == 2048


@pytest.mark.parametrize("version", [(0, 32, 1), (0, 33, 0)])
def test_unverified_mlx_keeps_the_dense_model_at_256(monkeypatch, version):
    monkeypatch.setattr(pd, "_mlx_version", lambda: version)
    assert pd.decayed_for_batch(_batch(_model(), 121_856)) == 512


def test_gpu_without_nax_keeps_the_dense_model_at_256(monkeypatch):
    monkeypatch.setattr(pd, "_nax_available", lambda: False)
    assert pd.decayed_for_batch(_batch(_model(), 121_856)) == 512
    assert pd.decayed_for_batch(
        _batch(_model(head_dim=128), 121_856)) == 2048


def test_fused_256_lever_keeps_the_dense_model(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_FUSED_256", "0")
    assert pd.decayed_for_batch(_batch(_model(), 121_856)) == 512


def test_cpu_default_device_keeps_the_dense_model(monkeypatch):
    monkeypatch.setattr(pd, "cpu_default_device", lambda: True)
    assert pd.decayed_for_batch(
        _batch(_model(head_dim=128), 2_000_000)) == 256


@pytest.mark.parametrize("cache", [None, []])
def test_no_cache_keeps_the_dense_model(cache):
    # Memory preflight prices a request before its cache exists.
    assert pd.resolve_score_profile(_model(), cache) is None


@pytest.mark.parametrize("cache", [None, []])
def test_no_cache_price_covers_the_cut_chunk_the_prompt_runs(cache):
    # An idle server, a 129,024-token prefix hit and a 913-token suffix:
    # the suffix runs whole, unfused, at 5.69 GB.
    runs = 24 * 913 * (129_024 + 913) * 2
    assert pd.score_transient_bytes(_model(), cache, 129_937) >= runs
    # Shallow, the dense price stays; 60 prefill rows' mask outweighs it.
    dense = 24 * 2048 * (10_000 + 2048) * 2
    assert pd.score_transient_bytes(_model(), cache, 10_000) == dense
    assert pd.score_transient_bytes(
        _model(), cache, 10_000, prefill_rows=60) == 61 * 2048 * 12_048
    assert pd.score_transient_bytes(
        _model(model_type="other_arch"), cache, 129_937) == (
        24 * 512 * (129_937 + 512) * 2)


def test_unregistered_arch_keeps_the_dense_model():
    assert pd.decayed_for_batch(
        _batch(_model(model_type="other_arch"), 121_856)) == 512


@pytest.mark.parametrize("extra", [
    {"global_head_dim": 512},
    {"kv_lora_rank": 512},
    {"q_lora_rank": 1536},
    {"qk_rope_head_dim": 64},
    {"v_head_dim": 128},
])
def test_unfused_or_split_attention_keeps_the_dense_model(extra):
    assert pd.decayed_for_batch(_batch(_model(**extra), 121_856)) == 512


@pytest.mark.parametrize("holder", ["args", "language_model"])
def test_mixed_head_dims_read_from_args_keep_the_dense_model(holder):
    # mlx-lm style models carry their config as args, with no config.
    args = SimpleNamespace(model_type="fused_arch", num_attention_heads=8,
                           head_dim=256, global_head_dim=512)
    m = (SimpleNamespace(args=args) if holder == "args"
         else SimpleNamespace(language_model=SimpleNamespace(args=args)))
    assert pd._attention_head_dims(m) == {256, 512}
    assert pd.fused_attention_profile(m, _caches(0)) is None


def test_unfused_head_dim_keeps_the_dense_model():
    assert pd.decayed_for_batch(
        _batch(_model(head_dim=512), 121_856)) == 512


def test_quantized_cache_keeps_the_dense_model():
    caches = _caches(121_856)
    caches[1].bits = 8
    assert pd.decayed_for_batch(
        _batch(_model(), 121_856, caches=caches)) == 512


def test_unknown_cache_class_keeps_the_dense_model():
    class OwnPrefillCache:  # e.g. KVarN, whose prefill materializes
        offset = 121_856

    caches = [OwnPrefillCache()]
    assert pd.decayed_for_batch(
        _batch(_model(), 121_856, caches=caches)) == 512


def test_kv_cache_subclass_keeps_the_dense_model():
    class OwnAttentionCache(kc.KVCache):  # e.g. a sparse-attention cache
        pass

    caches = _caches(121_856, kc.ArraysCache, OwnAttentionCache)
    assert pd.decayed_for_batch(
        _batch(_model(), 121_856, caches=caches)) == 512


def test_chunk_exactly_the_fused_width_stays_whole():
    # Dense, 1024 rows at 200k need 9.9 GB against the 5.7 GB cap.
    assert pd.decayed_for_batch(_batch(_model(), 200_000, base=1024)) == 1024


def test_tick_term_below_the_fused_width_refits_the_cap(monkeypatch):
    # The tick term picks 512, which runs unfused: 9.8 GB dense at 400k.
    monkeypatch.setattr(pd, "_LIVE_DECODE_ROWS", 1)
    monkeypatch.setattr(pd, "_CHUNK_COST", (2048, 1.5))
    assert pd.decayed_for_batch(_batch(_model(), 400_000)) == 256
    assert pd.decayed_for_batch(_batch(_model(), 100_000)) == 512


def test_base_below_the_fused_width_decays_as_before():
    # An explicit 512 chunk is not fused at 256, so the cap still applies.
    assert pd.decayed_for_batch(_batch(_model(), 200_000, base=512)) == 512
    assert pd.decayed_for_batch(
        _batch(_model(), 1_000_000, base=512)) == 256


def test_registered_arch_profile_wins():
    pd_flash = pd.ScoreTransientProfile(heads=1, bytes_per_elem=4)
    try:
        pd.register_score_profile("fused_arch_reg", lambda m, c: pd_flash)
        m = _model(model_type="fused_arch_reg")
        assert pd.resolve_score_profile(m, _caches(0)) == pd_flash
    finally:
        pd._SCORE_PROFILES.pop("fused_arch_reg", None)


def test_kill_switch_restores_the_dense_model(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_PROFILE", "0")
    assert pd.decayed_for_batch(_batch(_model(), 121_856)) == 512


def test_transient_prices_a_chunk_cut_under_the_fused_width():
    # A fused 2048 chunk holds nothing on a causal cache, but a cut chunk
    # runs unfused: 1023 rows whole at 20k, up to the cap at 200k.
    m = _model()
    assert pd.score_transient_bytes(m, _caches(20_000), 20_000) == (
        24 * 1023 * (20_000 + 1023) * 2)
    assert pd.score_transient_bytes(
        m, _caches(200_000), 200_000) == pd._body_cap_bytes()
    dense = pd.score_transient_bytes(m, [object()], 200_000)
    assert dense == 24 * 512 * (200_000 + 512) * 2


@pytest.mark.parametrize("dim,rows", [
    (64, 9), (72, 9), (80, 9), (96, 9), (128, 9), (256, 1024), (192, 0),
    (512, 0),
])
def test_fused_min_rows_table(dim, rows):
    assert pd.fused_min_rows(dim) == rows


@pytest.mark.parametrize("model_type", [
    "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"])
def test_qwen35_arms_for_its_language_model_only(model_type):
    import gmlx.models.qwen35  # noqa: F401
    from gmlx.models.qwen35.language import LanguageModel

    cfg = SimpleNamespace(model_type=model_type, num_attention_heads=24,
                          head_dim=256)
    target = SimpleNamespace(config=cfg,
                             language_model=LanguageModel.__new__(LanguageModel))
    profile = pd.resolve_score_profile(target, _caches(0))
    assert profile is not None and profile.fused_min_step == 1024
    # The MTP drafter shares the model type and has no language model.
    assert pd.resolve_score_profile(SimpleNamespace(config=cfg),
                                    _caches(0)) is None


# --- chunks cut after the step is chosen ---------------------------------


def _cut_batch(depth, cols, folds=False, **extra):
    m = _model()
    m.supports_logits_to_keep = folds
    b = _batch(m, depth)
    b.draft_model = None
    b._inputs_embeds = SimpleNamespace(shape=(1, cols, 8))
    b._processed_prompt_columns = 0
    b._right_pad_per_row = None
    b._next_apc_checkpoint_column = lambda: None
    b.__dict__.update(extra)
    return b


def _dense_fits(step, depth):
    return 24 * step * (depth + step) * 2 <= 5.7e9


def test_prompt_tail_under_the_fused_width_takes_the_dense_fit():
    # 975 rows at 129,024 need 6.08 GB unfused.
    assert pd.decayed_for_batch(_cut_batch(129_024, 976)) == 487
    assert pd.decayed_for_batch(_cut_batch(129_024, 976 + 2048)) == 2048
    assert pd.decayed_for_batch(_cut_batch(20_000, 976)) == 975


def test_folded_last_forward_under_the_fused_width_takes_the_dense_fit():
    # A follow-up turn with 700 new tokens at a 200k prefix-cache hit.
    assert pd.decayed_for_batch(_cut_batch(200_000, 700, folds=True)) == 350


@pytest.mark.parametrize("start", [0, 1000])
def test_checkpoint_and_row_end_cuts_take_the_dense_fit(start):
    b = _cut_batch(200_000, 5000, _processed_prompt_columns=start,
                   _next_apc_checkpoint_column=lambda: start + 700)
    assert pd.decayed_for_batch(b) == 350
    b = _cut_batch(200_000, 5000, _processed_prompt_columns=start,
                   _right_pad_per_row=[0, 4400],
                   _suffix_lens=[start + 600, 9000])
    assert pd.decayed_for_batch(b) == 300


def test_fit_cut_refits_only_a_fused_profile_under_its_width():
    b = _cut_batch(200_000, 5000)
    assert pd.fit_cut(b, 700) == 350
    assert pd.fit_cut(b, 1500) == 1500
    assert pd.fit_cut(b, 0) == 0
    b.model.config.model_type = "other_arch"
    assert pd.fit_cut(b, 700) == 700


def test_fit_cut_keeps_the_cut_of_a_profile_without_a_fused_width(
        monkeypatch):
    # e.g. a streamed MoE tail merge widens such a step past its fit.
    monkeypatch.setitem(pd._SCORE_PROFILES, "flash_arch",
                        lambda m, c: pd.ScoreTransientProfile(heads=24))
    b = _cut_batch(200_000, 5000)
    b.model.config.model_type = "flash_arch"
    assert pd.fit_cut(b, 2000) == 2000


def test_media_block_cut_under_the_fused_width_takes_the_dense_fit(
        monkeypatch):
    import gmlx.gen.media_spans as ms

    seen = []

    class Stub:
        _prompt_kwargs = {"image_spans": [[700, 3000]]}
        _inputs_embeds = SimpleNamespace(shape=(1, 5000, 8))
        _processed_prompt_columns = 0
        uids = [1]
        _apc_meta = None
        prefill_step_size = 2048
        model = _model()
        prompt_cache = _caches(200_000)

        def _next_apc_checkpoint_column(self):
            return None

        def prompt_step(self):
            seen.append(self.prefill_step_size)
            return "ok"

    monkeypatch.setattr(pd, "decayed_for_batch", lambda batch: 2048)
    assert ms.install_span_aware_prompt_step(Stub)
    assert Stub().prompt_step() == "ok"
    assert seen == [350]
    # A chunk the block widens keeps the whole block.
    seen.clear()
    Stub._prompt_kwargs = {"image_spans": [[0, 900]]}
    monkeypatch.setattr(pd, "decayed_for_batch", lambda batch: 350)
    assert Stub().prompt_step() == "ok"
    assert seen == [900]


def test_real_prompt_loop_runs_every_chunk_within_the_cap(monkeypatch):
    """Stock PromptProcessingBatch under the installed decay: every forward,
    the folded last one included, fits the cap at depth."""
    import mlx.core as mx
    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch as P

    monkeypatch.setattr(P, "prompt_step", P.prompt_step)
    monkeypatch.setattr(P, "needs_processing", P.needs_processing)
    monkeypatch.delattr(P, pd._FLAG, raising=False)
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert pd.install_prefill_decay()

    class Model:
        def __init__(self, folds):
            self.config = _model().config
            self.supports_logits_to_keep = folds
            self.calls = []

        def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
            b, n = input_ids.shape
            self.calls.append(n)
            kv = mx.zeros((b, 1, n, 4))
            for c in cache:
                c.update_and_fetch(kv, kv)
            return SimpleNamespace(logits=mx.zeros((b, n, 4)),
                                   hidden_states=[inputs_embeds],
                                   shared_kv_states=None)

    for depth, cols, folds in ((129_024, 976, False), (200_000, 700, True)):
        monkeypatch.setattr(pd, "kv_depth", lambda cache, d=depth: d)
        model = Model(folds)
        ids = list(range(cols))
        batch = P(model, [0], [ids], [4], mx.zeros((1, cols, 4)), {},
                  warm_cache=[kc.BatchKVCache(left_padding=[0])])
        while batch.needs_processing():
            batch.prompt_step()
        last = int(batch._inputs_embeds.shape[1])
        runs = model.calls + ([last] if folds else [])
        assert runs == ([350, 350] if folds else [487, 488])
        assert all(_dense_fits(n, depth) for n in runs), runs


def test_batch_cache_mask_bounds_a_fused_chunk():
    # B = 4 at 1M: the 2048 mask build holds 10.3 GB, 1024 holds 5.1 GB.
    caches = [kc.BatchKVCache(left_padding=[0, 0, 0, 0])]
    caches[0].offset = 1_000_000
    profile = pd.fused_attention_profile(_model(), caches)
    assert profile.fused_mask_bytes == 5
    assert pd.decayed_step(2048, 1_000_000, 24, profile=profile) == 1024
    price = pd.score_transient_bytes(_model(), caches, 1_000_000)
    assert price >= 5 * 1024 * (1_000_000 + 1024)
    assert pd.fused_attention_profile(
        _model(), _caches(0)).fused_mask_bytes == 0
    # 40 rows at 10k: the 1 GB mask outweighs a 0.54 GB cut chunk.
    wide = [kc.BatchKVCache(left_padding=[0] * 40)]
    wide[0].offset = 10_000
    assert pd.score_transient_bytes(_model(), wide, 10_000) == (
        41 * 2048 * (10_000 + 2048))


def test_mask_over_the_cap_keeps_the_fused_width_when_unfused_is_larger():
    profile = pd.ScoreTransientProfile(heads=24, bytes_per_elem=2,
                                       fused_min_step=1024,
                                       fused_mask_bytes=5)
    # 5 x 1024 x 2M = 10 GB fused, against 24.6 GB for the 256 floor.
    assert pd.decayed_step(2048, 2_000_000, 24, profile=profile) == 1024
    # With 48 rows the 1024 mask outweighs the unfused floor.
    big = profile._replace(fused_mask_bytes=49)
    assert pd.decayed_step(2048, 2_000_000, 24, profile=big) == 256


def test_nax_probe_failure_reads_as_no_nax(monkeypatch):
    import mlx_kquant

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(mlx_kquant, "nax_available", boom)
    assert _NAX_PROBE() is False


# --- the hd512 prefill tiler ------------------------------------------------


@pytest.mark.parametrize("mask", ["causal", "array", None])
def test_tiler_leaves_a_query_stock_fuses_whole(monkeypatch, mask):
    import mlx.core as mx

    import gmlx.device
    import gmlx.upstream.attn_hd512 as hd

    monkeypatch.setattr(gmlx.device, "cpu_default_device", lambda: False)
    q = mx.zeros((1, 2, 1024, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 4096, 256), dtype=mx.bfloat16)

    def m(rows=1024):
        if mask == "array":
            return mx.ones((rows, 4096), dtype=mx.bool_)
        return mask

    assert hd._prefill_eligible(q, k, k, m()) is (mask is None)
    assert hd._prefill_eligible(q[:, :, :1023], k, k, m(1023)) is True
    assert hd._prefill_eligible(q.astype(mx.float32), k.astype(mx.float32),
                                k.astype(mx.float32), m()) is True
    monkeypatch.setattr(pd, "_nax_available", lambda: False)
    assert hd._prefill_eligible(q, k, k, m()) is True
    monkeypatch.setattr(pd, "_nax_available", lambda: True)
    monkeypatch.setattr(gmlx.device, "cpu_default_device", lambda: True)
    assert hd._prefill_eligible(q, k, k, m()) is True


@pytest.mark.parametrize("spans,step,calls", [
    ([[700, 3000]], None, [350]),  # the block cuts the 2048 chunk to 700
    ([[0, 900]], 350, [900]),  # a widened chunk keeps the whole block
])
def test_mtp_body_media_cut_under_the_fused_width_takes_the_dense_fit(
        monkeypatch, spans, step, calls):
    import mlx.core as mx
    from gmlx._vendor.mlx_vlm.generate import ar

    import gmlx.spec.engine as engine
    import gmlx.spec.mtp_prefill as mtp_prefill

    mtp_prefill.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(engine, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", True)
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    monkeypatch.setattr(pd, "kv_depth", lambda cache: 200_000)
    if step:
        monkeypatch.setattr(pd, "decayed_for_batch", lambda batch: step)

    class Model:
        config = _model().config

        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
            b, n = input_ids.shape
            self.calls.append(n)
            kv = mx.zeros((b, 1, n, 4))
            for c in cache:
                c.update_and_fetch(kv, kv)
            return SimpleNamespace(logits=mx.zeros((b, n, 4)),
                                   hidden_states=[inputs_embeds],
                                   shared_kv_states=None)

    model = Model()
    ids = list(range(5000))
    batch = ar.PromptProcessingBatch(
        model, [0], [ids], [4], mx.zeros((1, 5000, 4)),
        {"image_spans": spans},
        warm_cache=[kc.BatchKVCache(left_padding=[0])],
        draft_model=object(), draft_kind="mtp")
    batch.prompt_step()
    assert model.calls == calls


@pytest.mark.parametrize("left_padding", [None, [0]])
def test_price_bounds_every_chunk_the_runtime_runs(left_padding):
    """The governor's price is at least the transient of any chunk the
    decayed loop runs, for every prompt tail at depth, folded or not."""
    m = _model()
    mask = 2 if left_padding else 0
    for depth in (60_000, 129_024, 262_144, 500_000, 1_000_000):
        if left_padding:
            caches = [kc.ArraysCache(1),
                      kc.BatchKVCache(left_padding=left_padding)]
            caches[1].offset = depth
        else:
            caches = _caches(depth)
        price = pd.score_transient_bytes(m, caches, depth)
        for cols in range(2, 4100, 37):
            for folds in (False, True):
                b = _cut_batch(depth, cols, folds=folds)
                b.prompt_cache = caches
                left = cols if folds else cols - 1
                n = min(pd.decayed_for_batch(b), left)
                cost = (mask * n * (depth + n) if n >= 1024
                        else 24 * n * (depth + n) * 2)
                assert cost <= price, (depth, cols, folds, n)


def test_non_power_of_two_step_halves_to_the_fused_width():
    profile = pd.ScoreTransientProfile(heads=24, bytes_per_elem=2,
                                       fused_min_step=1024,
                                       fused_mask_bytes=5)
    # 1536 rows hold 7.7 GB of mask at 1M; 1024 hold 5.1 GB.
    assert pd.decayed_step(1536, 1_000_000, 24, profile=profile) == 1024


def test_price_sizes_the_mask_from_the_prefill_rows():
    # 32 decode rows at 50k: their mask would be 3.5 GB; one prefill row
    # leaves the 2.5 GB cut chunk as the price.
    caches = [kc.BatchKVCache(left_padding=[0] * 32)]
    caches[0].offset = 50_000
    cut = 24 * 1023 * (50_000 + 1023) * 2
    assert pd.score_transient_bytes(_model(), caches, 50_000) == (
        33 * 2048 * (50_000 + 2048))
    assert pd.score_transient_bytes(
        _model(), caches, 50_000, prefill_rows=1) == cut
