#!/usr/bin/env python3
"""Tolerant GMLX_* env parsing. Several flags are read at import time, so
a malformed value (`GMLX_WALK_PROFILE=yes`) used to break `import gmlx`
or `gmlx serve` at boot; env_int/env_float fall back to the default instead.

Boolean flags (numerics kill-switches among them) route through env_bool; the
tests below pin each site's default AND that the disable value actually flips
the branch selection, so a kill-switch that does not kill cannot ship silently."""

from __future__ import annotations

import importlib
import inspect
import os

import mlx.core as mx
import pytest

from gmlx.envflags import env_bool, env_float, env_int


def test_env_int_unset_uses_default(monkeypatch):
    monkeypatch.delenv("GMLX_TEST_FLAG", raising=False)
    assert env_int("GMLX_TEST_FLAG", 7) == 7


def test_env_int_parses_and_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("GMLX_TEST_FLAG", "12")
    assert env_int("GMLX_TEST_FLAG", 7) == 12
    monkeypatch.setenv("GMLX_TEST_FLAG", "yes")
    assert env_int("GMLX_TEST_FLAG", 7) == 7
    monkeypatch.setenv("GMLX_TEST_FLAG", "")
    assert env_int("GMLX_TEST_FLAG", 7) == 7


def test_env_float_parses_and_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("GMLX_TEST_FLAG", "1.5")
    assert env_float("GMLX_TEST_FLAG", 30.0) == 1.5
    monkeypatch.setenv("GMLX_TEST_FLAG", "soon")
    assert env_float("GMLX_TEST_FLAG", 30.0) == 30.0


# ---------------------------------------------------------------------------
# env_bool


@pytest.mark.parametrize("raw", ["1", "true", "True", "YES", "on", " on "])
def test_env_bool_truthy(monkeypatch, raw):
    monkeypatch.setenv("GMLX_TEST_FLAG", raw)
    assert env_bool("GMLX_TEST_FLAG", False) is True
    assert env_bool("GMLX_TEST_FLAG", True) is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "OFF", ""])
def test_env_bool_falsy(monkeypatch, raw):
    monkeypatch.setenv("GMLX_TEST_FLAG", raw)
    assert env_bool("GMLX_TEST_FLAG", True) is False
    assert env_bool("GMLX_TEST_FLAG", False) is False


@pytest.mark.parametrize("default", [True, False])
def test_env_bool_unset_and_garbage_use_default(monkeypatch, default):
    monkeypatch.delenv("GMLX_TEST_FLAG", raising=False)
    assert env_bool("GMLX_TEST_FLAG", default) is default
    monkeypatch.setenv("GMLX_TEST_FLAG", "garbage")
    assert env_bool("GMLX_TEST_FLAG", default) is default


# ---------------------------------------------------------------------------
# Import-time flag sites: reload the module under a pinned env and check the
# branch-selecting constant. The fixture restores env and reloads once more so
# later tests see the environment-default constants.


@pytest.fixture
def flagmod():
    saved = {}
    touched = []
    # attn_hd512 patches this global at install time; reloads must not leave
    # a stale wrapper (or a wrapped wrapper) behind for later tests.
    sdpa = mx.fast.scaled_dot_product_attention

    def _load(modname, **env):
        for k, v in env.items():
            saved.setdefault(k, os.environ.get(k))
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if modname not in touched:
            touched.append(modname)
        return importlib.reload(importlib.import_module(modname))

    yield _load
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for m in touched:
        importlib.reload(importlib.import_module(m))
    # Restore the TRUE original, not a previously-installed wrapper: the
    # reload re-executed attn_hd512's module dict (_orig_sdpa = None), so a
    # restored stale wrapper would fall back to None on its stock route.
    # Leaving the patch uninstalled is safe -- installs are idempotent and
    # re-run on every _install_and_load.
    mx.fast.scaled_dot_product_attention = getattr(
        sdpa, "_gmlx_orig_sdpa", sdpa)


_HD512_FLAGS = [
    # (env name, module constant, unset default)
    ("GMLX_HD256_VERIFY", "_HD256_VERIFY", True),
    ("GMLX_GQA_SDPA", "_GQA_DECODE", True),
    ("GMLX_GQA_SDPA_HD256", "_GQA_HD256", False),
    ("GMLX_VERIFY_GEMM", "_VERIFY_GEMM", True),
    ("GMLX_VERIFY_FA", "_VERIFY_FA", True),
]


@pytest.mark.parametrize("name,attr,default", _HD512_FLAGS)
def test_attn_hd512_flag_defaults_and_flip(flagmod, name, attr, default):
    mod = flagmod("gmlx.attn_hd512", **{name: None})
    assert getattr(mod, attr) is default
    flip = "0" if default else "1"
    mod = flagmod("gmlx.attn_hd512", **{name: flip})
    assert getattr(mod, attr) is (not default)
    # word forms now accepted
    word = "off" if default else "on"
    mod = flagmod("gmlx.attn_hd512", **{name: word})
    assert getattr(mod, attr) is (not default)


def test_hd512_install_kill_switch(flagmod):
    import mlx.core as mx

    orig = mx.fast.scaled_dot_product_attention
    try:
        mod = flagmod("gmlx.attn_hd512", GMLX_HD512="0")
        assert mod.install_hd512_sdpa() is False
        assert mx.fast.scaled_dot_product_attention is orig
        mod = flagmod("gmlx.attn_hd512", GMLX_HD512=None)
        assert mod.install_hd512_sdpa() is True
        assert mx.fast.scaled_dot_product_attention is not orig
    finally:
        mx.fast.scaled_dot_product_attention = orig


def test_speculative_flag_defaults(flagmod):
    mod = flagmod(
        "gmlx.speculative",
        GMLX_MTP_COUPLED_DRAFT=None,
        GMLX_ROUND_PROFILE=None,
        GMLX_MTP_TOP2_LOG=None,
        GMLX_SPEC_APC=None,
        GMLX_SPEC_APC_SIDECAR=None,
    )
    assert mod._FORCE_GREEDY_DRAFT is True
    assert mod._ROUND_PROFILE is False
    assert mod._TOP2_LOG is False
    assert mod._SIDECAR_DISABLED is False


def test_speculative_flag_flips(flagmod):
    mod = flagmod("gmlx.speculative", GMLX_MTP_COUPLED_DRAFT="1")
    assert mod._FORCE_GREEDY_DRAFT is False
    mod = flagmod(
        "gmlx.speculative",
        GMLX_MTP_COUPLED_DRAFT=None,
        GMLX_ROUND_PROFILE="1",
        GMLX_MTP_TOP2_LOG="1",
    )
    assert mod._FORCE_GREEDY_DRAFT is True
    assert mod._ROUND_PROFILE is True
    assert mod._TOP2_LOG is True


@pytest.mark.parametrize("name", ["GMLX_SPEC_APC", "GMLX_SPEC_APC_SIDECAR"])
def test_speculative_sidecar_kill_switch(flagmod, name):
    mod = flagmod(
        "gmlx.speculative",
        GMLX_ROUND_PROFILE=None,
        GMLX_MTP_TOP2_LOG=None,
        **{name: "0"},
    )
    assert mod._SIDECAR_DISABLED is True


def test_mtp_drafter_postnorm_feed_flag(flagmod):
    mod = flagmod("gmlx.mtp_drafter", GMLX_MTP_POSTNORM_FEED=None)
    assert mod._POSTNORM_FEED is False
    mod = flagmod("gmlx.mtp_drafter", GMLX_MTP_POSTNORM_FEED="1")
    assert mod._POSTNORM_FEED is True


# ---------------------------------------------------------------------------
# Call-time flag sites in loader.py: exercise the patch functions directly
# with stub models; no weights are loaded.


class _Boom(Exception):
    pass


class _NoModules:
    def modules(self):
        return []


class _BoomModules:
    def modules(self):
        raise _Boom


def test_fused_gdn_kill_switch_decode(monkeypatch):
    patches = importlib.import_module("gmlx.gdn_patches")
    if patches._gdn_fused_decode_kernel is None:
        pytest.skip("fused gdn decode kernel unavailable")
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    saved_call = GatedDeltaNet.__call__
    saved_installed = patches._FUSED_DECODE_PATCH.installed
    saved_stock = patches._FUSED_DECODE_PATCH.stock
    patches._FUSED_DECODE_PATCH.installed = False
    try:
        monkeypatch.setenv("GMLX_FUSED_GDN", "0")
        patches._patch_gated_delta_fused_decode(_NoModules())
        assert patches._FUSED_DECODE_PATCH.installed is False
        assert GatedDeltaNet.__call__ is saved_call  # unfused branch kept
        monkeypatch.delenv("GMLX_FUSED_GDN")
        patches._patch_gated_delta_fused_decode(_NoModules())
        assert patches._FUSED_DECODE_PATCH.installed is True
        assert GatedDeltaNet.__call__ is not saved_call
    finally:
        GatedDeltaNet.__call__ = saved_call
        patches._FUSED_DECODE_PATCH.installed = saved_installed
        patches._FUSED_DECODE_PATCH.stock = saved_stock


def test_fused_gdn_kill_switch_verify(monkeypatch):
    patches = importlib.import_module("gmlx.gdn_patches")
    if patches._gdn_fused_verify_kernel is None:
        pytest.skip("fused gdn verify kernel unavailable")
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    saved_call = Qwen3_5GatedDeltaNet.__call__
    saved_installed = patches._FUSED_VERIFY_PATCH.installed
    saved_stock = patches._FUSED_VERIFY_PATCH.stock
    patches._FUSED_VERIFY_PATCH.installed = False
    try:
        monkeypatch.setenv("GMLX_FUSED_GDN", "0")
        patches._patch_gated_delta_fused_verify(_NoModules())
        assert patches._FUSED_VERIFY_PATCH.installed is False
        assert Qwen3_5GatedDeltaNet.__call__ is saved_call
        monkeypatch.delenv("GMLX_FUSED_GDN")
        patches._patch_gated_delta_fused_verify(_NoModules())
        assert patches._FUSED_VERIFY_PATCH.installed is True
        assert Qwen3_5GatedDeltaNet.__call__ is not saved_call
    finally:
        Qwen3_5GatedDeltaNet.__call__ = saved_call
        patches._FUSED_VERIFY_PATCH.installed = saved_installed
        patches._FUSED_VERIFY_PATCH.stock = saved_stock


def test_gdn_zba_default_off_and_opt_in(monkeypatch):
    patches = importlib.import_module("gmlx.gdn_patches")
    if patches._gdn_fused_decode_kernel is None:
        pytest.skip("fused gdn decode kernel unavailable")
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    saved_call = GatedDeltaNet.__call__
    saved_installed = patches._FUSED_DECODE_PATCH.installed
    saved_stock = patches._FUSED_DECODE_PATCH.stock
    merged = []
    monkeypatch.setattr(
        patches, "_gdn_try_merge_zba", lambda m: bool(merged.append(m)))
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)  # isinstance target, no weights

    class _One:
        def modules(self):
            return [gdn]

    try:
        monkeypatch.delenv("GMLX_GDN_ZBA", raising=False)
        patches._patch_gated_delta_fused_decode(_One())
        assert not merged  # default off
        monkeypatch.setenv("GMLX_GDN_ZBA", "1")
        patches._patch_gated_delta_fused_decode(_One())
        assert len(merged) == 1
    finally:
        GatedDeltaNet.__call__ = saved_call
        patches._FUSED_DECODE_PATCH.installed = saved_installed
        patches._FUSED_DECODE_PATCH.stock = saved_stock


_DSV32_KILL_SITES = [
    # (env name, patch fn, ClassPatch global, patched class)
    ("GMLX_DSV32_MASK_DECODE", "_patch_dsv32_mask_decode",
     "_MASK_DECODE_PATCH", "DeepseekV32Attention"),
    ("GMLX_DSV32_INDEXER_ROPE", "_patch_dsv32_indexer_rope", None, None),
    ("GMLX_DSV32_INDEXER_FP32", "_patch_dsv32_indexer_fp32",
     "_INDEXER_FP32_PATCH", "Indexer"),
    ("GMLX_DSV32_GATE_FP32", "_patch_dsv32_moe_gate_fp32",
     "_GATE_FP32_PATCH", "MoEGate"),
]


@pytest.mark.parametrize("name,fn,flag,clsname", _DSV32_KILL_SITES)
def test_dsv32_kill_switches(monkeypatch, name, fn, flag, clsname):
    patches = importlib.import_module("gmlx.dsv32_patches")
    dsv32 = importlib.import_module("mlx_lm.models.deepseek_v32")
    patch = getattr(patches, fn)
    cls = getattr(dsv32, clsname) if clsname else None
    saved_call = cls.__call__ if cls else None
    cp = getattr(patches, flag) if flag else None
    saved_installed = cp.installed if cp else None
    if cp:
        cp.installed = False
    try:
        # kill value: returns before touching the model
        monkeypatch.setenv(name, "0")
        patch(_BoomModules())
        if cp:
            assert cp.installed is False
        # default on: reaches the model walk
        monkeypatch.delenv(name)
        with pytest.raises(_Boom):
            patch(_BoomModules())
    finally:
        if cls:
            cls.__call__ = saved_call
        if cp:
            cp.installed = saved_installed


def test_dsv32_sparse_opt_in(monkeypatch):
    patches = importlib.import_module("gmlx.dsv32_patches")
    warns = []
    monkeypatch.setattr(patches.loadlog, "warn", lambda *a, **k: warns.append(a))
    # default (unset): dense-default patch walks the model, no warning
    monkeypatch.delenv("GMLX_DSV32_SPARSE", raising=False)
    with pytest.raises(_Boom):
        patches._patch_dsv32_dense_default(_BoomModules())
    assert not warns
    # opt-in: warns and returns, leaving the native sparse indexer alone
    monkeypatch.setenv("GMLX_DSV32_SPARSE", "1")
    patches._patch_dsv32_dense_default(_BoomModules())
    assert warns


def test_f16_head_kernel_kill_switch(monkeypatch):
    patches = importlib.import_module("gmlx.gdn_patches")
    if patches._F16_HEAD_GEMV is None:
        pytest.skip("f16 head gemv kernel unavailable")

    class _BoomHead:
        @property
        def language_model(self):
            raise _Boom

    monkeypatch.setenv("GMLX_F16_HEAD_KERNEL", "0")
    patches._patch_dense_head_verify(_BoomHead())  # returns before the model
    monkeypatch.delenv("GMLX_F16_HEAD_KERNEL")
    with pytest.raises(_Boom):
        patches._patch_dense_head_verify(_BoomHead())


# ---------------------------------------------------------------------------
# GMLX_GEMMA_FUSED_GLU is genuinely tri-state (1 = force on, 0 = force off,
# unset/other = codec-gated auto), so it stays a raw read; pin the shape and
# the unset default so a refactor to env_bool cannot silently drop the auto arm.


def test_gemma_fused_glu_tristate_unset_default():
    modules = importlib.import_module("gmlx.modules")
    src = inspect.getsource(modules)
    assert 'os.environ.get("GMLX_GEMMA_FUSED_GLU", "")' in src
    assert 'env == "1"' in src  # force-on arm
    assert 'env != "0"' in src  # auto arm (codec-gated) unless forced off


def test_gdn_zba_merge_content_and_load_order():
    loader = importlib.import_module("gmlx.loader")
    patches = importlib.import_module("gmlx.gdn_patches")
    if patches._gdn_fused_decode_kernel is None:
        pytest.skip("fused gdn decode kernel unavailable")
    import inspect

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    gdn = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn.value_dim = 8
    gdn.num_v_heads = 2
    gdn.in_proj_z = nn.Linear(4, 8, bias=False)
    gdn.in_proj_b = nn.Linear(4, 2, bias=False)
    gdn.in_proj_a = nn.Linear(4, 2, bias=False)
    wz, wb, wa = (gdn.in_proj_z.weight, gdn.in_proj_b.weight,
                  gdn.in_proj_a.weight)
    assert patches._gdn_try_merge_zba(gdn)
    expect = mx.concatenate([wz, wb, wa], axis=0)
    assert mx.array_equal(gdn._gdn_zba_weight, expect)
    assert mx.array_equal(gdn.in_proj_z.weight, wz)
    assert mx.array_equal(gdn.in_proj_b.weight, wb)
    assert mx.array_equal(gdn.in_proj_a.weight, wa)

    # The merge snapshots weights, so it must run after load_weights and the
    # kquant leaf swap; merging at the runtime-patch step would bake the
    # constructor's random init into _gdn_zba_weight.
    src = inspect.getsource(loader.load_model)
    assert src.index("model.load_weights(") < src.index(
        "_patch_gated_delta_fused_decode(")


def test_ignore_eos_gate_is_a_bool_flag(monkeypatch):
    """`GMLX_IGNORE_EOS=0` must disable the benchmark mode. The gate used
    `os.environ.get(...)`, whose '0' is truthy, so opting out turned it on and
    every completion ran to max_tokens."""
    from gmlx import server

    src = inspect.getsource(server._serve)
    assert 'os.environ.get("GMLX_IGNORE_EOS")' not in src
    assert 'env_bool("GMLX_IGNORE_EOS", False)' in src

    monkeypatch.setenv("GMLX_IGNORE_EOS", "0")
    assert env_bool("GMLX_IGNORE_EOS", False) is False
    monkeypatch.setenv("GMLX_IGNORE_EOS", "1")
    assert env_bool("GMLX_IGNORE_EOS", False) is True
