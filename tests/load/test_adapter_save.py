#!/usr/bin/env python3
"""GGUF LoRA adapter WRITER (train T1): ``save_lora_adapter`` is the inverse of the
P1 loader. Round-trips synthetic A/B - including a q/k pair that exercises the
forward qk-permute - through a real temp GGUF file and back via
``load_lora_adapter``, asserting identity. CPU-only (tiny f32 arrays; the wire
reader runs on CPU)."""
from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("gguf")
from gmlx import adapter  # noqa: E402
from gmlx.transforms import qk_permute_wire  # noqa: E402

# llama-arch geometry: 4 q-heads, 2 kv-heads, head_dim 8 -> q_out 32, k_out 16.
N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 8
IN, R, ALPHA = 16, 4, 16.0
Q_OUT = N_HEAD * HEAD_DIM      # 32
K_OUT = N_HEAD_KV * HEAD_DIM   # 16


def _ab(out, in_, r, *, seed):
    mx.random.seed(seed)
    return mx.random.normal((r, in_)), mx.random.normal((out, r))


def _modules():
    qa, qb = _ab(Q_OUT, IN, R, seed=1)
    ka, kb = _ab(K_OUT, IN, R, seed=2)
    da, db = _ab(IN, IN, R, seed=3)   # an mlp (down_proj) - passthrough, no permute
    return [
        ("model.layers.0.self_attn.q_proj", qa, qb),
        ("model.layers.0.self_attn.k_proj", ka, kb),
        ("model.layers.0.mlp.down_proj", da, db),
    ]


def _save(tmp_path, modules, **kw):
    path = str(tmp_path / "adapter.gguf")
    n = adapter.save_lora_adapter(
        path, modules, alpha=ALPHA, base_arch="llama",
        n_head=N_HEAD, n_head_kv=N_HEAD_KV, n_layers=2, **kw)
    return path, n


def test_save_roundtrips_through_loader(tmp_path):
    modules = _modules()
    path, n = _save(tmp_path, modules)
    assert n == 3

    plan = adapter.load_lora_adapter(path)
    assert plan.alpha == ALPHA
    assert plan.arch == "llama"
    assert set(plan.modules) == {m[0] for m in modules}

    for module_path, a, b in modules:
        lm = plan.modules[module_path]
        assert lm.rank == R
        assert lm.scale == pytest.approx(ALPHA / R)
        assert mx.allclose(lm.a, a.astype(mx.float32))
        if lm.transform == "qk_permute":
            nh = N_HEAD_KV if module_path.endswith("k_proj") else N_HEAD
            # writer forward-permuted b; the loader's de-permute recovers the input
            assert mx.allclose(qk_permute_wire(lm.b, nh), b.astype(mx.float32))
        else:
            assert mx.allclose(lm.b, b.astype(mx.float32))


def test_qk_modules_tagged_permute_mlp_passthrough(tmp_path):
    plan = adapter.load_lora_adapter(_save(tmp_path, _modules())[0])
    assert plan.modules["model.layers.0.self_attn.q_proj"].transform == "qk_permute"
    assert plan.modules["model.layers.0.self_attn.k_proj"].transform == "qk_permute"
    assert plan.modules["model.layers.0.mlp.down_proj"].transform == "passthrough"


def test_unknown_arch_raises(tmp_path):
    qa, qb = _ab(Q_OUT, IN, R, seed=1)
    with pytest.raises(ValueError, match="MODEL_ARCH"):
        adapter.save_lora_adapter(
            str(tmp_path / "a.gguf"),
            [("model.layers.0.self_attn.q_proj", qa, qb)],
            alpha=ALPHA, base_arch="not_an_arch",
            n_head=N_HEAD, n_head_kv=N_HEAD_KV, n_layers=2)


def test_empty_modules_raises(tmp_path):
    with pytest.raises(ValueError, match="no LoRA modules"):
        adapter.save_lora_adapter(
            str(tmp_path / "a.gguf"), [], alpha=ALPHA, base_arch="llama",
            n_head=N_HEAD, n_head_kv=N_HEAD_KV, n_layers=2)
