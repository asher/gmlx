"""Expert LoRA on fused MoE blocks, on real weights. A rank-1 adapter minted
at test time adapts the expert and shared-expert down projections of
layers 0 and 1. In rows mode a decode step with the slot off gives the base
logits bit for bit. With the slot on, each adapted block matches the same
block with the fused blocks off, at the input it saw in that step, and the
logits move from the base. On the MTP target of a qwen35moe file, a verify
block applies the adapter as a plain forward over the same tokens does.

The per-block checks are the claim. Each adapted block also matches the
stock leaf with the plain-op delta (``_FUSED_MOE_ENABLED`` off), so the
check does not rest on the fused epilogue alone. The logits of the whole
step with the fused blocks off are a sanity check only: the stock router
rounds its scores to bfloat16 and flips an expert at a near tie on some
unadapted layers, where the fused router keeps float32. Their bound is the
same step's gap with the slot off plus half the move, so it fails when the
delta is skipped, but not when one adapted layer is.

``integration``; skips unless ``KQUANT_TEST_GGUF_DIR`` holds a qwen35moe
GGUF, or a qwen4exp GGUF (Qwen3.8-Flash-Next) for the qwen4exp row.
"""

from __future__ import annotations

import gc
import os

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
# Relative L2 bounds, set on the first runs (M3 Max, qwen35moe Q6_K and
# IQ3_M). An adapted block against the same block with the fused blocks
# off, at the decode step's input: measured 0.0096 at most; against the
# stock leaf: 0.0146 at most.
BLOCK_TOL = 0.02
# The step's logits against the step with the fused blocks off may differ
# by the gap with the slot off plus this share of the move: measured 0.041,
# against a gap of 0.020 and a move of 0.092.
MOVE_SHARE = 0.5
# The least move of the logits from the base with the slot on: measured
# 0.092 (plain) and 0.129 (verify).
MIN_MOVE = 0.05
# A verify block's logits against a plain forward over the same tokens:
# measured 0.027 with the slot on and 0.021 with it off. An adapted owned
# block, verify against plain, at the verify input: measured 0.0045.
VERIFY_TOL = 0.05



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


def _qwen4exp(gguf_index):
    """A qwen4exp GGUF (Qwen3.8-Flash-Next) whose shards are all present."""
    from gmlx.load.preflight import find_split_shards

    for path in gguf_index.get("qwen4exp", ()):
        try:
            find_split_shards(path)
        except FileNotFoundError:
            continue
        return path
    pytest.skip("no complete 'qwen4exp' GGUF under KQUANT_TEST_GGUF_DIR")


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
            gap = _rel(base, step(0))
        move, err = _rel(on, base), _rel(on, ref)
        print(f"[{arch}] logits: slot on vs base {move:.4f}, vs blocks off "
              f"{err:.4f}, gap with the slot off {gap:.4f}")
        assert move > MIN_MOVE
        assert err < gap + MOVE_SHARE * move
        for b in adapted:
            x = seen[id(b)]
            with lora_rows.bind(reg), lora_rows.published([1]):
                got = _np(b(x))
                with monkeypatch.context() as m:
                    m.setattr(modules, "_FUSED_MOE_BLOCK_ENABLED", False)
                    want = _np(b(x))
                with monkeypatch.context() as m:
                    m.setattr(modules, "_FUSED_MOE_ENABLED", False)
                    stock = _np(b(x))
            with lora_rows.bind(reg), lora_rows.published([0]):
                bare = _np(b(x))
            print(f"[{arch}] block: vs blocks off {_rel(got, want):.4f}, "
                  f"vs stock {_rel(got, stock):.4f}, move {_rel(got, bare):.4f}")
            assert _rel(got, want) < BLOCK_TOL
            assert _rel(got, stock) < BLOCK_TOL
            assert _rel(got, bare) > 10 * BLOCK_TOL
    finally:
        _free(raw)


def test_qwen35moe_plain_load_folds_off_and_matches_unfused_on(
        gguf_index, tmp_path, monkeypatch):
    _fused_block_check(_qwen35moe(gguf_index), tmp_path, monkeypatch)


def test_qwen4exp_folds_off_and_matches_unfused_on(gguf_index, tmp_path,
                                                    monkeypatch):
    _fused_block_check(_qwen4exp(gguf_index), tmp_path, monkeypatch)


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
