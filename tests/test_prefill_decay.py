"""Depth-decay prefill chunk scaling: tier math + batch resolution."""

from types import SimpleNamespace

from gmlx import prefill_decay as pd

HEADS = 32
GB = 1e9


def _cap(monkeypatch, gb):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", str(gb))


def _transient(step, depth, heads=HEADS):
    return heads * step * (depth + step) * 2


def test_shallow_keeps_base(monkeypatch):
    _cap(monkeypatch, 1.0)
    # 32 * 2048 * 2048 * 2 = 268 MB fits 1 GB
    assert pd.decayed_step(2048, 0, HEADS) == 2048


def test_decay_tiers_with_depth(monkeypatch):
    _cap(monkeypatch, 1.0)
    seen = []
    for depth in (0, 8_192, 32_768, 100_000, 200_000):
        step = pd.decayed_step(2048, depth, HEADS)
        seen.append(step)
        # chosen tier fits the cap unless already at the floor
        if step > 256:
            assert _transient(step, depth) <= 1.0 * GB
        # the next tier up would NOT fit (largest-fitting property)
        if step < 2048:
            assert _transient(step * 2, depth) > 1.0 * GB
    # monotone non-increasing in depth
    assert seen == sorted(seen, reverse=True)
    assert seen[0] == 2048 and seen[-1] == 256


def test_floor_holds_at_extreme_depth(monkeypatch):
    _cap(monkeypatch, 0.5)
    assert pd.decayed_step(2048, 10_000_000, HEADS) == 256


def test_explicit_small_base_passes_through(monkeypatch):
    _cap(monkeypatch, 0.5)
    # an explicit PREFILL_STEP_SIZE at/below the floor stays authoritative
    assert pd.decayed_step(128, 10_000_000, HEADS) == 128
    assert pd.decayed_step(256, 10_000_000, HEADS) == 256


def test_min_step_env(monkeypatch):
    _cap(monkeypatch, 0.5)
    monkeypatch.setenv("GMLX_PREFILL_MIN_STEP", "512")
    assert pd.decayed_step(2048, 10_000_000, HEADS) == 512


def test_idempotent(monkeypatch):
    _cap(monkeypatch, 1.0)
    for depth in (0, 50_000, 200_000):
        once = pd.decayed_step(2048, depth, HEADS)
        assert pd.decayed_step(once, depth, HEADS) == once


def test_none_and_zero_base_pass_through(monkeypatch):
    _cap(monkeypatch, 1.0)
    assert pd.decayed_step(0, 1000, HEADS) == 0
    assert pd.decayed_step(None, 1000, HEADS) is None


def test_kv_depth():
    caches = [SimpleNamespace(offset=100), SimpleNamespace(offset=4096),
              SimpleNamespace(), SimpleNamespace(offset=None)]
    assert pd.kv_depth(caches) == 4096
    assert pd.kv_depth([]) == 0
    assert pd.kv_depth(None) == 0


def test_score_heads():
    m = SimpleNamespace(config=SimpleNamespace(num_attention_heads=24))
    assert pd.score_heads(m) == 24
    nested = SimpleNamespace(config=SimpleNamespace(
        num_attention_heads=None,
        text_config=SimpleNamespace(num_attention_heads=32)))
    assert pd.score_heads(nested) == 32
    assert pd.score_heads(SimpleNamespace(config=SimpleNamespace())) == 32
    # config objects work directly too (drafter path)
    assert pd.score_heads(SimpleNamespace(num_attention_heads=16)) == 16


def _batch(base, depth, heads=HEADS):
    return SimpleNamespace(
        prefill_step_size=base,
        prompt_cache=[SimpleNamespace(offset=depth)],
        model=SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=heads)),
    )


def test_decayed_for_batch(monkeypatch):
    _cap(monkeypatch, 1.0)
    assert pd.decayed_for_batch(_batch(2048, 0)) == 2048
    assert pd.decayed_for_batch(_batch(2048, 100_000)) == 256
    assert pd.decayed_for_batch(_batch(None, 100_000)) is None


def test_kill_switch(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_for_batch(_batch(2048, 100_000)) == 2048
    assert pd.install_prefill_decay() is False


def _headroom(monkeypatch, val):
    monkeypatch.setattr(pd, "_headroom_bytes", lambda: val)


def test_seed_follows_body_cap_when_headroom_unknown(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, None)
    for depth in (0, 32_768, 200_000):
        assert (pd.decayed_seed_step(2048, depth, HEADS)
                == pd.decayed_step(2048, depth, HEADS))


def test_seed_pins_base_when_headroom_large(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 90 * GB)  # seed cap = 45 GB
    depth = 200_000
    # 32 * 2048 * ~202k * 2 = 26.5 GB fits half the 90 GB headroom
    assert pd.decayed_seed_step(2048, depth, HEADS) == 2048
    # the body still decays under its own 1 GB cap
    assert pd.decayed_step(2048, depth, HEADS) == 256


def test_seed_decays_under_tight_headroom(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 30 * GB)  # seed cap = 15 GB
    # 1024 tier (13.2 GB) fits, 2048 (26.5 GB) does not
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 1024
    # negative headroom floors at the body cap
    _headroom(monkeypatch, -5 * GB)
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 256


def test_seed_cap_env_wins_over_headroom(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, 2 * GB)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "30")
    # 32 * 2048 * ~202k * 2 = 26.5 GB fits the explicit 30 GB cap
    assert pd.decayed_seed_step(2048, 200_000, HEADS) == 2048


def test_seed_cap_env_invalid_falls_back(monkeypatch):
    _cap(monkeypatch, 1.0)
    _headroom(monkeypatch, None)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "banana")
    assert (pd.decayed_seed_step(2048, 200_000, HEADS)
            == pd.decayed_step(2048, 200_000, HEADS))


def test_seed_step_honors_kill_switch(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_seed_step(2048, 10_000_000, HEADS) == 2048


def test_note_untracked_weights_accumulates(monkeypatch):
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", 0.0)
    pd.note_untracked_weights(10 * GB)
    pd.note_untracked_weights(5 * GB)
    assert pd._UNTRACKED_WEIGHTS == 15 * GB


# --- arch score profiles -------------------------------------------------

FLASH = pd.ScoreTransientProfile(heads=1, bytes_per_elem=4, depth_divisor=4)


def _register(monkeypatch, model_type, provider):
    monkeypatch.setitem(pd._SCORE_PROFILES, model_type, provider)


def _pbatch(base, depth, model_type="fake_arch", heads=64):
    return SimpleNamespace(
        prefill_step_size=base,
        prompt_cache=[SimpleNamespace(offset=depth)],
        model=SimpleNamespace(config=SimpleNamespace(
            model_type=model_type, num_attention_heads=heads)),
    )


def test_profile_holds_full_step_at_depth(monkeypatch):
    _cap(monkeypatch, 5.0)
    # flash-native transient [1, step, depth/4] fp32: full step to ~2.4M
    assert pd.decayed_step(2048, 200_000, 64, profile=FLASH) == 2048
    assert pd.decayed_step(2048, 1_000_000, 64, profile=FLASH) == 2048
    assert pd.decayed_step(2048, 3_000_000, 64, profile=FLASH) == 1024
    # dense model at the same cap floors out
    assert pd.decayed_step(2048, 200_000, 64) == 256


def test_dense_equivalent_profile_matches_default(monkeypatch):
    _cap(monkeypatch, 1.0)
    dense = pd.ScoreTransientProfile(heads=HEADS, bytes_per_elem=2,
                                     depth_divisor=1)
    for depth in (0, 8_192, 32_768, 100_000, 200_000):
        assert (pd.decayed_step(2048, depth, HEADS, profile=dense)
                == pd.decayed_step(2048, depth, HEADS))


def test_profile_floor_and_passthrough(monkeypatch):
    _cap(monkeypatch, 0.5)
    heavy = pd.ScoreTransientProfile(heads=64, bytes_per_elem=4,
                                     depth_divisor=1)
    assert pd.decayed_step(2048, 10_000_000, 64, profile=heavy) == 256
    # explicit base at/below the floor still passes through untouched
    assert pd.decayed_step(128, 10_000_000, 64, profile=heavy) == 128


def test_profile_idempotent(monkeypatch):
    _cap(monkeypatch, 5.0)
    for depth in (0, 200_000, 3_000_000):
        once = pd.decayed_step(2048, depth, 64, profile=FLASH)
        assert pd.decayed_step(once, depth, 64, profile=FLASH) == once


def test_profile_cap_env_still_authoritative(monkeypatch):
    _cap(monkeypatch, 0.1)  # the env floor
    # a tiny explicit cap forces decay even under a flash profile
    assert pd.decayed_step(2048, 300_000, 64, profile=FLASH) == 256


def test_resolve_score_profile_registry(monkeypatch):
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    m = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"))
    assert pd.resolve_score_profile(m, None) == FLASH
    other = SimpleNamespace(config=SimpleNamespace(model_type="other"))
    assert pd.resolve_score_profile(other, None) is None
    # dict-shaped configs resolve too (CLI/MTP wrapper paths)
    md = SimpleNamespace(config={"model_type": "fake_arch"})
    assert pd.resolve_score_profile(md, None) == FLASH
    assert pd.resolve_score_profile(SimpleNamespace(), None) is None


def test_resolve_score_profile_provider_failures(monkeypatch):
    _register(monkeypatch, "none_arch", lambda m, c: None)
    _register(monkeypatch, "raise_arch",
              lambda m, c: (_ for _ in ()).throw(RuntimeError("boom")))
    none_m = SimpleNamespace(config=SimpleNamespace(model_type="none_arch"))
    raise_m = SimpleNamespace(config=SimpleNamespace(model_type="raise_arch"))
    assert pd.resolve_score_profile(none_m, None) is None
    assert pd.resolve_score_profile(raise_m, None) is None


def test_resolve_score_profile_kill_switch(monkeypatch):
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    monkeypatch.setenv("GMLX_PREFILL_SCORE_PROFILE", "0")
    m = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"))
    assert pd.resolve_score_profile(m, None) is None


def test_decayed_for_batch_with_profile(monkeypatch):
    _cap(monkeypatch, 1.0)
    _register(monkeypatch, "fake_arch", lambda m, c: FLASH)
    # profile holds the full step where the dense model would floor out
    assert pd.decayed_for_batch(_pbatch(2048, 100_000)) == 2048
    unregistered = _pbatch(2048, 100_000, model_type="plain_arch")
    assert pd.decayed_for_batch(unregistered) == 256


def test_profile_base_step_swap(monkeypatch):
    _cap(monkeypatch, 5.0)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    with_base = FLASH._replace(base_step=4096)
    _register(monkeypatch, "fake_arch", lambda m, c: with_base)
    # stock 2048 base swaps to the arch default at any depth it fits
    assert pd.decayed_for_batch(_pbatch(2048, 0)) == 4096
    assert pd.decayed_for_batch(_pbatch(2048, 200_000)) == 4096
    # an explicit PREFILL_STEP_SIZE keeps the swap off
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert pd.decayed_for_batch(_pbatch(2048, 0)) == 2048
    monkeypatch.delenv("PREFILL_STEP_SIZE")
    # a non-stock batch base is explicit config: no swap
    assert pd.decayed_for_batch(_pbatch(1024, 0)) == 1024
    # the kill switch bypasses swap and decay alike
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.decayed_for_batch(_pbatch(2048, 200_000)) == 2048


def test_decayed_seed_step_with_profile(monkeypatch):
    _cap(monkeypatch, 1.0)
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 256
    assert pd.decayed_seed_step(2048, 200_000, 64, profile=FLASH) == 2048


# --- provider builder ------------------------------------------------------

class _Pool(SimpleNamespace):
    pass


class _BatchPool(SimpleNamespace):
    pass


def _built(**kw):
    kw.setdefault("profile", FLASH)
    kw.setdefault("require_cache", _Pool)
    kw.setdefault("disarm_cache", _BatchPool)
    return pd.build_score_profile(**kw)


def _layer(*caches):
    return SimpleNamespace(caches=list(caches))


def test_walk_caches_flattens_cachelist():
    a, b, c = (SimpleNamespace(offset=i) for i in range(3))
    assert list(pd._walk_caches([_layer(a, b), c])) == [a, b, c]
    assert list(pd._walk_caches(None)) == []


def test_build_kernels_gate():
    armed = {"on": False}
    prov = _built(kernels_armed=lambda: armed["on"])
    cache = [_layer(_Pool(offset=100))]
    assert prov(None, cache) is None
    armed["on"] = True
    assert prov(None, cache) == FLASH


def test_build_require_cache_missing_returns_none():
    prov = _built()
    assert prov(None, [SimpleNamespace(offset=100)]) is None
    assert prov(None, None) is None
    assert prov(None, [_layer(_Pool(offset=100))]) == FLASH


def test_build_disarm_cache():
    prov = _built()
    assert prov(None, [_layer(_Pool(offset=1), _BatchPool(offset=1))]) is None


def test_build_non_int_offset_disarms():
    prov = _built()
    assert prov(None, [_layer(_Pool(offset=[3, 5]))]) is None


def test_build_quantized_required_disarms_by_default():
    qpool = _Pool(offset=100, is_quantized=True)
    assert _built()(None, [_layer(qpool)]) is None
    assert _built(allow_quantized_pools=True)(None, [_layer(qpool)]) == FLASH


def test_build_bits_on_other_cache_disarms():
    # a quantized non-required cache (local KV with bits) always disarms,
    # even when quantized required pools are allowed
    prov = _built(allow_quantized_pools=True)
    cache = [_layer(SimpleNamespace(offset=1, bits=8), _Pool(offset=100))]
    assert prov(None, cache) is None


def test_build_callable_profile_resolved_per_call():
    live = {"base": None}
    prov = _built(profile=lambda: FLASH._replace(base_step=live["base"]))
    cache = [_layer(_Pool(offset=100))]
    assert prov(None, cache).base_step is None
    live["base"] = 4096
    assert prov(None, cache).base_step == 4096


# --- seed profile threading + logging ---------------------------------------

def test_seed_profile_resolves_against_drafter_cache(monkeypatch):
    from gmlx.mtp_drafter import QwenMTPDrafter
    prov = pd.build_score_profile(profile=FLASH, require_cache=_Pool)
    _register(monkeypatch, "fake_arch", prov)
    drafter = SimpleNamespace(config=SimpleNamespace(model_type="fake_arch"),
                              _cache=[SimpleNamespace(offset=7)])
    # a plain-KV drafter cache has no required pool: dense seed decay stays
    assert QwenMTPDrafter._seed_profile(drafter) is None
    drafter._cache = [_Pool(offset=7)]
    assert QwenMTPDrafter._seed_profile(drafter) == FLASH


def test_seed_decay_log(monkeypatch, capsys):
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1")
    monkeypatch.setenv("GMLX_PREFILL_DECAY_LOG", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 256
    out = capsys.readouterr().out
    assert "[prefill-decay] seed depth 200000: step 2048 -> 256" in out


def test_seed_decay_log_silent_when_unchanged(monkeypatch, capsys):
    monkeypatch.setenv("GMLX_MTP_SEED_SCORE_CAP_GB", "1000")
    monkeypatch.setenv("GMLX_PREFILL_DECAY_LOG", "1")
    assert pd.decayed_seed_step(2048, 200_000, 64) == 2048
    assert capsys.readouterr().out == ""


# --- headroom-aware body cap -------------------------------------------------

GIB = float(1 << 30)


def _hcap(monkeypatch, *, gate=True, noted=8 * GIB, room=55 * GB,
          legacy=6.0 * GB):
    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB", raising=False)
    monkeypatch.setenv("GMLX_PREFILL_HEADROOM_CAP", "1" if gate else "0")
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", noted)
    monkeypatch.setattr(pd, "_headroom_bytes", lambda: room)
    monkeypatch.setattr(pd, "_WS_CAP_BYTES", legacy)


def test_headroom_cap_default_off(monkeypatch):
    _hcap(monkeypatch, gate=True)
    monkeypatch.delenv("GMLX_PREFILL_HEADROOM_CAP")
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_requires_noted_limit(monkeypatch):
    _hcap(monkeypatch, noted=None)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_raises_at_depth(monkeypatch):
    # the 122B shape: 32 heads, 55 GB free, 8 GiB cache limit. Raised cap
    # = min(27.5 GB, 8.59 GB) = 8.59 GB: 1024 at d110k (legacy 512) and 512
    # at d200k (legacy 256); held transients (7.3 / 6.6 GB) stay under the
    # recycle threshold.
    _hcap(monkeypatch, gate=False)
    assert pd.decayed_step(2048, 110_000, 32) == 512
    assert pd.decayed_step(2048, 200_000, 32) == 256
    _hcap(monkeypatch, gate=True)
    assert pd._body_cap_bytes() == 8 * GIB
    assert pd.decayed_step(2048, 110_000, 32) == 1024
    assert pd.decayed_step(2048, 200_000, 32) == 512


def test_headroom_cap_env_cap_still_wins(monkeypatch):
    _hcap(monkeypatch)
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "1.0")
    assert pd._body_cap_bytes() == 1.0 * GB


def test_headroom_cap_probe_failure_falls_back(monkeypatch):
    _hcap(monkeypatch, room=None)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_headroom_cap_never_below_legacy(monkeypatch):
    # tight headroom on a limit-configured box: legacy floor holds
    _hcap(monkeypatch, room=4 * GB)
    assert pd._body_cap_bytes() == 6.0 * GB


def test_note_cache_limit_none_clears(monkeypatch):
    monkeypatch.setattr(pd, "_NOTED_CACHE_LIMIT", None)
    pd.note_cache_limit(8 * GIB)
    assert pd._NOTED_CACHE_LIMIT == 8 * GIB
    pd.note_cache_limit(None)
    assert pd._NOTED_CACHE_LIMIT is None


def test_seed_cap_unchanged_by_gate(monkeypatch):
    # seed = max(body cap, 0.5*headroom); with ample headroom the seed cap is
    # 0.5h either way -- the certified seed formula is untouched by the gate
    _hcap(monkeypatch, gate=False, room=90 * GB)
    monkeypatch.delenv("GMLX_MTP_SEED_SCORE_CAP_GB", raising=False)
    off = pd.decayed_seed_step(2048, 10_000_000, 64)
    _hcap(monkeypatch, gate=True, room=90 * GB)
    assert pd.decayed_seed_step(2048, 10_000_000, 64) == off


def test_headroom_cap_no_noted_limit_matches_legacy_ladder(monkeypatch):
    _hcap(monkeypatch, gate=True, noted=None)
    for depth in (0, 8_192, 46_000, 92_000, 110_000, 200_000, 400_000):
        with_gate = pd.decayed_step(2048, depth, 32)
        _hcap(monkeypatch, gate=False, noted=None)
        assert with_gate == pd.decayed_step(2048, depth, 32)
        _hcap(monkeypatch, gate=True, noted=None)
