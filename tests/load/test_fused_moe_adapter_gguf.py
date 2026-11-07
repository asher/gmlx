"""Expert LoRA on fused MoE blocks, on real weights. A rank-1 adapter minted
at test time adapts the expert and shared-expert down projections of
layers 0 and 1. In rows mode a decode step with the slot off gives the base
logits bit for bit. With the slot on, each adapted block matches the same
block with the fused blocks off, at the input it saw in that step, and the
logits move from the base. On the MTP target of a qwen35moe file, a verify
block applies the adapter as a plain forward over the same tokens does.

The logits of the whole step with the fused blocks off are a loose
reference only: the stock router rounds its scores to bfloat16 and flips an
expert at a near tie on some unadapted layers, where the fused router
keeps float32.

``integration``; skips unless ``KQUANT_TEST_GGUF_DIR`` holds a qwen35moe
GGUF, or the Qwen3.8-Flash-Next UD-IQ4_XS shards for the qwen4exp row.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from helpers import mint_lora_gguf

from gmlx import lora_rows
import gmlx.load.modules as modules

pytestmark = pytest.mark.integration

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused MoE kernels need a real GPU", allow_module_level=True)

PROMPT = "The capital of France is Paris. The capital of Italy is"
LAYERS = (0, 1)
# Relative L2 bounds, set on the first runs (M3 Max, qwen35moe Q6_K). An
# adapted block against the same block with the fused blocks off, at the
# decode step's input: measured 0.0096 at most.
BLOCK_TOL = 0.02
# The step's logits against the step with the fused blocks off: measured
# 0.041, against a floor of 0.020 with the slot off.
LOGITS_TOL = 0.08
# The least move of the logits from the base with the slot on: measured
# 0.092 (plain) and 0.129 (verify).
MIN_MOVE = 0.05
# A verify block's logits against a plain forward over the same tokens:
# measured 0.027 with the slot on and 0.021 with it off. An adapted owned
# block, verify against plain, at the verify input: measured 0.0045.
VERIFY_TOL = 0.05

FLASH_NEXT = "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
FLASH_NEXT_BYTES = 93_682_584_224


@pytest.fixture(autouse=True)
def _rows():
    lora_rows.configure("rows", 1)
    yield
    lora_rows.configure("static", 1)


def _qwen35moe(gguf_index):
    paths = gguf_index.get("qwen35moe")
    if not paths:
        pytest.skip("no 'qwen35moe' GGUF under KQUANT_TEST_GGUF_DIR")
    return paths[0]


def _flash_next(gguf_index):
    hit = [p for p in gguf_index.get("qwen4exp", ()) if Path(p).name == FLASH_NEXT]
    if not hit:
        pytest.skip(f"no {FLASH_NEXT} under KQUANT_TEST_GGUF_DIR")
    shards = [Path(hit[0]).with_name(FLASH_NEXT.replace("00001-of", f"0000{i}-of"))
              for i in (1, 2, 3)]
    if not all(p.is_file() for p in shards) or \
            sum(p.stat().st_size for p in shards) != FLASH_NEXT_BYTES:
        pytest.skip("the Flash-Next UD-IQ4_XS shards are missing or of another size")
    return hit[0]


def _mint(gguf, tmp_path):
    """A rank-1 adapter on the expert and shared-expert down projections of
    LAYERS, sized from the base's headers."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    arch = scan_gguf(gguf, include_tensors=False).kv["general.architecture"]
    shapes = {t.name: t.shape for shard in find_split_shards(gguf)
              for t in scan_gguf(shard).tensors}
    rng = np.random.default_rng(0)
    pairs = {}
    for i in LAYERS:
        inter, hidden, e = shapes[f"blk.{i}.ffn_down_exps.weight"]
        pairs[f"blk.{i}.ffn_down_exps.weight"] = (
            rng.standard_normal((e, 1, inter)) / np.sqrt(inter),
            rng.standard_normal((e, hidden, 1)) * 0.5)
        s_inter, hidden = shapes[f"blk.{i}.ffn_down_shexp.weight"]
        pairs[f"blk.{i}.ffn_down_shexp.weight"] = (
            rng.standard_normal((1, s_inter)) / np.sqrt(s_inter),
            rng.standard_normal((hidden, 1)) * 0.5)
    out = tmp_path / "expert.lora.gguf"
    mint_lora_gguf(out, arch, pairs)
    return str(out), arch


def _registry():
    reg = lora_rows.UidRegistry()
    reg.register(0, (0.0,))
    reg.register(1, (1.0,))
    return reg


def _logits(out):
    return getattr(out, "logits", out)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _np(a):
    a = a.astype(mx.float32)
    mx.eval(a)
    return np.array(a)


def _run(lm, cache, prompt, block, reg, uid, **kw):
    """Prefill ``prompt``, then run ``block``, each forward published for
    ``uid``; the block's logits as float32."""
    with lora_rows.bind(reg):
        with lora_rows.published([uid]):
            mx.eval(_logits(lm(mx.array([prompt]), cache=cache)))
        with lora_rows.published([uid]):
            return _np(_logits(lm(mx.array([block]), cache=cache, **kw))[0])


def _capture(monkeypatch, cls, width):
    """Record the input of every ``cls`` call of ``width`` tokens."""
    seen = {}
    orig = cls.__call__

    def rec(self, x, *a, **k):
        if x.size // x.shape[-1] == width:
            seen.setdefault(id(self), x)
        return orig(self, x, *a, **k)

    monkeypatch.setattr(cls, "__call__", rec)
    return seen


def _fold_calls(monkeypatch):
    """Record each call of the fold kernel."""
    import mlx_kquant as kq

    seen = []
    real = kq.moe_glu_gather_shexp_kq

    def spy(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(kq, "moe_glu_gather_shexp_kq", spy)
    return seen


def _free(*objs):
    del objs
    gc.collect()
    mx.clear_cache()


def _fused_block_check(gguf, tmp_path, monkeypatch):
    from gmlx import load_model
    from gmlx._vendor.mlx_lm.models.cache import make_prompt_cache
    from gmlx.load.adapter import apply_gguf_adapter

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    adapter_path, arch = _mint(gguf, tmp_path)
    raw, config, tok = load_model(gguf, verbose=False)
    try:
        blocks = [m for _, m in raw.named_modules()
                  if type(m).__name__ == "_FusedKQuantMoeBlock"]
        if not blocks:
            pytest.skip("no fused MoE block on this load")
        before = {id(b): (type(b), b.__dict__.get("_kq_shexp_fold", True))
                  for b in blocks}
        ids = list(tok.encode(PROMPT))
        reg = _registry()

        def step(uid):
            return _run(raw, make_prompt_cache(raw), ids[:-1], ids[-1:], reg,
                        uid)[-1]

        base = step(0)
        apply_gguf_adapter(raw, config, adapter_path, base_arch=arch)
        adapted = [b for b in blocks if b.__dict__.get("_kq_lora_slots")]
        assert len(adapted) == len(LAYERS)
        assert all(b._kq_expert_lora for b in adapted)
        for b in blocks:
            assert (type(b), b.__dict__.get("_kq_shexp_fold", True)) == before[id(b)]
        folds = _fold_calls(monkeypatch)
        assert np.array_equal(step(0), base)
        n_off = len(folds)
        folds.clear()

        with monkeypatch.context() as m:
            seen = _capture(m, type(blocks[0]), 1)
            on = step(1)
        # Every fold block folds at the decode step with the slot off; with
        # it on, the adapted ones call their members.
        fold_ids = {id(b) for b in blocks if b.__dict__.get("_kq_shexp_fold", True)
                    and 1 not in b.__dict__.get("_kq_fold_skip_t", ())}
        print(f"[{arch}] {len(blocks)} fused blocks, {len(fold_ids)} fold")
        assert n_off == len(fold_ids)
        assert len(folds) == n_off - sum(id(b) in fold_ids for b in adapted)
        with monkeypatch.context() as m:
            m.setattr(modules, "_FUSED_MOE_BLOCK_ENABLED", False)
            ref = step(1)
        move, err = _rel(on, base), _rel(on, ref)
        print(f"[{arch}] logits: slot on vs base {move:.4f}, vs blocks off {err:.4f}")
        assert move > MIN_MOVE
        assert err < LOGITS_TOL
        for b in adapted:
            x = seen[id(b)]
            with lora_rows.bind(reg), lora_rows.published([1]):
                got = _np(b(x))
                with monkeypatch.context() as m:
                    m.setattr(modules, "_FUSED_MOE_BLOCK_ENABLED", False)
                    want = _np(b(x))
            with lora_rows.bind(reg), lora_rows.published([0]):
                bare = _np(b(x))
            print(f"[{arch}] block: vs blocks off {_rel(got, want):.4f}, "
                  f"move {_rel(got, bare):.4f}")
            assert _rel(got, want) < BLOCK_TOL
            assert _rel(got, bare) > 10 * BLOCK_TOL
    finally:
        _free(raw)


def test_qwen35moe_plain_load_folds_off_and_matches_unfused_on(
        gguf_index, tmp_path, monkeypatch):
    _fused_block_check(_qwen35moe(gguf_index), tmp_path, monkeypatch)


def test_flash_next_folds_off_and_matches_unfused_on(gguf_index, tmp_path,
                                                     monkeypatch):
    _fused_block_check(_flash_next(gguf_index), tmp_path, monkeypatch)


def test_qwen35moe_mtp_verify_applies_the_expert_adapter(gguf_index, tmp_path,
                                                         monkeypatch):
    from gmlx.load.adapter import apply_gguf_adapter
    from gmlx.load.headerscan import scan_gguf
    from gmlx.spec.mtp_load import load_mtp_model

    gguf = _qwen35moe(gguf_index)
    if not scan_gguf(gguf, include_tensors=False).kv.get(
            "qwen35moe.nextn_predict_layers"):
        pytest.skip("this qwen35moe GGUF has no native MTP head")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    adapter_path, arch = _mint(gguf, tmp_path)
    installs = []

    def install(trunk, config):
        installs.append(apply_gguf_adapter(trunk, config, adapter_path,
                                           base_arch=arch))

    model, drafter, _config, tok = load_mtp_model(gguf, wire=False,
                                                  before_wire=install)
    try:
        assert installs and installs[0] > 0
        lm = model.language_model
        ids = list(tok.encode(PROMPT))
        prompt, block = ids[:-3], ids[-3:]
        reg = _registry()
        verify_kw = {"capture_layer_ids": [], "return_hidden": True,
                     "return_shared_kv": True}
        adapted = [m for _, m in lm.named_modules()
                   if type(m).__name__ == "OwnedQwen3_5MoeSparseMoeBlock"
                   and getattr(m.switch_mlp.down_proj, "_kq_lora", None) is not None]
        assert len(adapted) == len(LAYERS)
        with monkeypatch.context() as m:
            seen = _capture(m, type(adapted[0]), len(block))
            verify = _run(lm, lm.make_cache(), prompt, block, reg, 1, **verify_kw)
        plain = _run(lm, lm.make_cache(), prompt, block, reg, 1)
        base = _run(lm, lm.make_cache(), prompt, block, reg, 0, **verify_kw)
        move, err = _rel(verify, base), _rel(verify, plain)
        print(f"[{arch} mtp] logits: verify vs base {move:.4f}, vs plain {err:.4f}")
        assert move > MIN_MOVE
        assert err < VERIFY_TOL
        for b in adapted:
            x = seen[id(b)]
            with lora_rows.bind(reg), lora_rows.published([1]):
                got = _np(b(x, target_verify=True))
                want = _np(b(x, target_verify=False))
            with lora_rows.bind(reg), lora_rows.published([0]):
                bare = _np(b(x, target_verify=True))
            print(f"[{arch} mtp] block: verify vs plain {_rel(got, want):.4f}, "
                  f"move {_rel(got, bare):.4f}")
            assert _rel(got, want) < BLOCK_TOL
            assert _rel(got, bare) > 10 * BLOCK_TOL
    finally:
        _free(model, drafter)
