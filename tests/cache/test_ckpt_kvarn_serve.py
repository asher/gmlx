"""Ckpt tier x kvarn on the serve path: the shared rebuild predicate, the
in-place single-stream conversion, warm-hit clobber guards, live per-batch
layout signatures, and the production wrap-chain order.

CPU-only: real cache instances throughout (the layout tag and the
conversion both key on concrete classes, so stubs would never fire them);
constructors are pure Python. The wrap-chain test runs in a subprocess so
the install order it pins is the production one, not whatever this
process's earlier tests left behind.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.spec.engine as spec_engine
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.compat import runtime_cache_module
from gmlx.cache.snapshot import ckpt_lookup, ckpt_store
from gmlx.cache.kvarn_cache import BatchKVarNKVCache, KVarNKVCache
from gmlx.cache.kvarn_serve import _ppb_rebuild_declined, ensure_ppb_kvarn

from test_ckpt_kvarn import KVARN_TAG, _arr, _hollow_kvarn
from kvarn_testlib import Args

_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache

KW = {"kv_quant_scheme": "kvarn"}


class _HybridLM:
    def __init__(self, head_dim=128):
        self.args = Args(head_dim=head_dim)
        self.layers = [object()] * 2

    def make_cache(self):
        return [KVCache(), ArraysCache(size=2)]


def _batch(model=None, caches=None, uids=(0,), cols=0):
    return SimpleNamespace(
        model=model or _HybridLM(),
        uids=list(uids),
        prompt_cache=(caches if caches is not None
                      else [KVCache(), ArraysCache(size=2)]),
        _processed_prompt_columns=cols,
    )


def test_predicate_clean_batch_allows(kvarn_ops_ok):
    assert _ppb_rebuild_declined(_batch(), dict(KW)) is None


def test_predicate_kwarg_declines(kvarn_ops_ok):
    for kw, reason in [
        ({}, "scheme"),
        ({**KW, "kv_bits": 6}, "kv_bits"),
        ({**KW, "warm_cache": object()}, "warm_cache"),
        ({**KW, "draft_model": object()}, "draft"),
        ({**KW, "right_pad_per_row": [0]}, "right_pad"),
    ]:
        assert _ppb_rebuild_declined(_batch(), kw) == reason
    assert _ppb_rebuild_declined(_batch(uids=(0, 1)), dict(KW)) == "batch"


def test_predicate_converted_and_warm_decline(kvarn_ops_ok):
    b = _batch(caches=[KVarNKVCache(), ArraysCache(size=2)])
    assert _ppb_rebuild_declined(b, dict(KW)) == "converted"
    b = _batch(caches=[BatchKVarNKVCache(left_padding=[0]),
                       ArraysCache(size=2)])
    assert _ppb_rebuild_declined(b, dict(KW)) == "converted"
    # A populated or trimmed batch adopted a warm cache after
    # construction; the rebuild must never replace it.
    kv = KVCache()
    kv.offset = 5
    b = _batch(caches=[kv, ArraysCache(size=2)])
    assert _ppb_rebuild_declined(b, dict(KW)) == "populated"
    assert _ppb_rebuild_declined(_batch(cols=3), dict(KW)) == "trimmed"


def test_predicate_config_declines(kvarn_ops_ok, monkeypatch):
    # Config problems decline at the predicate, so B=1 keeps the stock
    # single-stream caches (fp16 tiers keep working) instead of being
    # rebuilt into fp16 batch classes.
    b = _batch(model=_HybridLM(head_dim=64))
    assert _ppb_rebuild_declined(b, dict(KW)) == "unsupported"
    monkeypatch.setenv("KV_BITS", "7")
    assert _ppb_rebuild_declined(_batch(), dict(KW)) == "unsupported"
    monkeypatch.delenv("KV_BITS")
    monkeypatch.setenv("KV_TAIL_TOKENS", "100")
    assert _ppb_rebuild_declined(_batch(), dict(KW)) == "unsupported"


def test_ensure_converts_in_place(kvarn_ops_ok, monkeypatch, capsys):
    from gmlx.cache import kvarn_serve

    monkeypatch.setattr(kvarn_serve, "_CKPT_NOTED", [False])
    monkeypatch.setenv("KV_TAIL_TOKENS", "256")
    b = _batch()
    arr = b.prompt_cache[1]
    assert ensure_ppb_kvarn(b, dict(KW), ckpt_active=True) is True
    assert type(b.prompt_cache[0]) is KVarNKVCache
    assert b.prompt_cache[0].tail_cap == 256
    assert b.prompt_cache[1] is arr
    assert "[kv] serve ckpt:" in capsys.readouterr().out
    # The outer batch rebuild now declines: the conversion cannot be
    # clobbered by the later wrap in the init chain.
    assert _ppb_rebuild_declined(b, dict(KW)) == "converted"


def test_ensure_gates(kvarn_ops_ok):
    b = _batch()
    kv = b.prompt_cache[0]
    assert ensure_ppb_kvarn(b, dict(KW), ckpt_active=False) is False
    assert b.prompt_cache[0] is kv
    assert ensure_ppb_kvarn(b, {}, ckpt_active=True) is False
    assert b.prompt_cache[0] is kv


def test_ensure_zero_conversion_leaves_stock(kvarn_ops_ok):
    # recurrent_gemma-shaped stack: no plain KV layer anywhere, so the
    # conversion is a no-op and the stock fp16 ckpt tier keeps working.
    caches = [ArraysCache(size=2), RotatingKVCache(max_size=32)]
    b = _batch(caches=list(caches))
    assert ensure_ppb_kvarn(b, dict(KW), ckpt_active=True) is False
    assert b.prompt_cache[0] is caches[0]
    assert b.prompt_cache[1] is caches[1]


def test_layout_live_ignores_model_memo():
    # Converted-ness is per-request: the model-cached probe must never
    # override what the live caches actually are.
    model = SimpleNamespace(_kq_apc_ckpt_layout=("kv", "arr"))
    b = SimpleNamespace(model=model,
                        prompt_cache=[KVarNKVCache(), ArraysCache(size=2)])
    assert spec_engine._ckpt_layout_live(b) == (KVARN_TAG, "arr")
    b2 = SimpleNamespace(model=model,
                         prompt_cache=[KVCache(), ArraysCache(size=2)])
    assert spec_engine._ckpt_layout_live(b2) == ("kv", "arr")


def test_layout_live_falls_back_to_model_probe():
    model = SimpleNamespace(_kq_apc_ckpt_layout=("kv", "arr"))
    b = SimpleNamespace(model=model, prompt_cache=None)
    assert spec_engine._ckpt_layout_live(b) == ("kv", "arr")


def test_layout_live_unsupported_refuses_all_records():
    b = SimpleNamespace(
        model=SimpleNamespace(),
        prompt_cache=[BatchKVarNKVCache(left_padding=[0]),
                      ArraysCache(size=2)],
    )
    sig = spec_engine._ckpt_layout_live(b)
    assert sig == spec_engine._LAYOUT_UNSUPPORTED
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    ids = list(range(500, 532))
    assert ckpt_store(man, ids, [_hollow_kvarn(32), _arr()], extra_hash=0)
    warm, got = ckpt_lookup(man, ids + [999], extra_hash=0, layout=sig)
    assert warm is None and got == 0


def test_store_lookup_signature_agreement_live():
    # The signature a live batch derives is the one its own stores key.
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    p = 32
    ids = list(range(500, 500 + p))
    b = SimpleNamespace(model=SimpleNamespace(),
                        prompt_cache=[_hollow_kvarn(p), _arr(seed=p)])
    assert ckpt_store(man, ids, b.prompt_cache, extra_hash=0)
    warm, got = ckpt_lookup(man, ids + [999], extra_hash=0,
                            layout=spec_engine._ckpt_layout_live(b))
    assert got == p and type(warm[0]) is KVarNKVCache


def _plain_batch(man, ids, caches):
    n = len(ids)
    meta = {"full_input_ids": list(ids), "prefix_len": 0, "extra_hash": 0,
            "apc_blocks": [], "checkpoint_len": max(0, n - 4)}
    return SimpleNamespace(
        model=SimpleNamespace(_kq_apc_ckpt=True, config=SimpleNamespace()),
        uids=[0],
        _apc_manager=man,
        _apc_mode="exact",
        _apc_meta=[meta],
        _right_pad_per_row=None,
        _inputs_embeds=mx.zeros((1, n, 4)),
        _input_ids=mx.array([ids]),
        _prompt_kwargs={},
        _prompt_length_aware_keys=[],
        _processed_prompt_columns=0,
        prefill_step_size=16,
        prompt_cache=caches,
        _apc_harvest_enabled=True,
    )


def test_plain_init_adopts_kvarn_record():
    # End to end on the stock path: a kvarn-boot batch's live signature
    # matches a kvarn record and the warm adoption trims the prompt.
    spec_engine._bind_l1_view()
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = list(range(300, 396))
    assert ckpt_store(man, ids[:32], [_hollow_kvarn(32), _arr(seed=5)],
                      extra_hash=0)
    b = _plain_batch(man, ids, [KVarNKVCache(), ArraysCache(size=2)])
    spec_engine._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 32
    assert type(b.prompt_cache[0]) is KVarNKVCache
    assert b.prompt_cache[0].offset == 32
    assert b._kq_ckpt_armed and b._apc_harvest_enabled is False


def test_plain_init_stock_batch_refuses_kvarn_record():
    # Same model, stock caches (the conversion declined this request):
    # the live signature refuses the kvarn record instead of adopting.
    spec_engine._bind_l1_view()
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = list(range(300, 396))
    assert ckpt_store(man, ids[:32], [_hollow_kvarn(32), _arr(seed=5)],
                      extra_hash=0)
    b = _plain_batch(man, ids, [KVCache(), ArraysCache(size=2)])
    spec_engine._plain_ckpt_init(b)
    assert b._processed_prompt_columns == 0
    assert b._kq_ckpt_armed


def test_production_wrap_chain_order():
    pytest.importorskip("mlx_vlm.generate.ar")
    # Install order restated from install_server_patches (spec engine
    # first, kvarn second): the kvarn wraps must be outermost with the
    # spec engine's inits directly inside, on all three chains.
    script = r"""
import importlib
# importlib, not `import mlx_vlm.generate.ar as ...`: the package exports
# a `generate` function that shadows the submodule attribute.
ar = importlib.import_module("mlx_vlm.generate.ar")
import gmlx.spec.engine as spec_engine
from gmlx.cache import kvarn_serve as ks
spec_engine.install_full_prompt_mtp_prefill()
ks.install_kvarn_serve()

def closure_names(fn):
    out = set()
    for cell in fn.__closure__ or ():
        try:
            out.add(getattr(cell.cell_contents, "__name__", None))
        except ValueError:
            pass
    return out

init = ar.PromptProcessingBatch.__init__
assert getattr(init, ks._PPB_FLAG, False), "kvarn PPB wrap not outermost"
assert "_mtp_init" in closure_names(init), "spec init not inside kvarn wrap"
bg = ar.BatchGenerator.__init__
assert getattr(bg, ks._APC_GATE_FLAG, False), "kvarn APC gate not outermost"
assert "_init_with_stash" in closure_names(bg), "stash not inside APC gate"
assert getattr(ar._make_cache, ks._MAKE_CACHE_FLAG, False)
print("CHAIN-OK")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=dict(os.environ), timeout=300)
    assert r.returncode == 0, r.stderr
    assert "CHAIN-OK" in r.stdout


def test_production_disk_arm_chain_order():
    """The APC disk arms chain, they do not replace each other. Install
    order restated from install_server_patches: pooling, then QSA, then
    kvarn. kvarn must be outermost and a non-kvarn record must still reach
    the QSA arm underneath it."""
    pytest.importorskip("mlx_vlm.apc")
    script = r"""
import importlib
apc = importlib.import_module("mlx_vlm.apc")
from gmlx.cache.apc_pooling import install_pooling_apc_support
from gmlx.cache.apc_qsa import install_qsa_apc_support
from gmlx.cache.kvarn_apc import install_kvarn_apc

NAMES = ("_snapshot_exact_cache_entry", "_load_exact_cache_entry")

def arms():
    return tuple(getattr(apc.DiskBlockStore, n) for n in NAMES)

install_pooling_apc_support()
pooled = arms()
install_qsa_apc_support()
qsa = arms()
install_kvarn_apc()
kvarn = arms()

def chained(fn):
    out = []
    for cell in fn.__closure__ or ():
        try:
            out.append(cell.cell_contents)
        except ValueError:
            pass
    return out

for i, name in enumerate(NAMES):
    assert kvarn[i] is not qsa[i], name       # kvarn installed its arm
    assert qsa[i] is not pooled[i], name      # QSA installed its own
    # kvarn chains the arm it found -- QSA's -- not the bare upstream.
    assert any(c is qsa[i] for c in chained(kvarn[i])), name
    assert any(c is pooled[i] for c in chained(qsa[i])), name
print("DISK-CHAIN-OK")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=dict(os.environ), timeout=300)
    assert r.returncode == 0, r.stderr
    assert "DISK-CHAIN-OK" in r.stdout
