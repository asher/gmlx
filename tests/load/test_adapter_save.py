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
import gmlx.load.adapter as adapter  # noqa: E402
from gmlx.load.transforms import qk_permute_wire  # noqa: E402

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


def _save(tmp_path, modules, *, base_arch="llama", **kw):
    path = str(tmp_path / "adapter.gguf")
    n = adapter.save_lora_adapter(
        path, modules, alpha=ALPHA, base_arch=base_arch,
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


# Two LoRA targets for each architecture the loader reads and gguf-py has
# no tensor-name map for, as (module path, base tensor, output width).
_OWNED = {
    "deepseek4": [("model.layers.0.attn.wq_a", "blk.0.attn_q_a.weight", IN),
                  ("model.layers.1.attn.wo_b", "blk.1.attn_output_b.weight", IN)],
    "deepseek41": [("model.layers.0.attn.wkv", "blk.0.attn_kv.weight", IN),
                   ("model.layers.1.ffn.shared_experts.down_proj", "blk.1.ffn_down_shexp.weight", IN)],
    "diffusion-gemma": [("model.decoder.layers.0.self_attn.q_proj", "blk.0.attn_q.weight", IN),
                        ("model.decoder.layers.1.mlp.down_proj", "blk.1.ffn_down.weight", IN)],
    "glm5next": [("model.layers.0.self_attn.q_proj", "blk.0.attn_q.weight", IN),
                 ("model.layers.1.self_attn.f_a_proj", "blk.1.ssm_f_a.weight", IN)],
    "hy_v3": [("model.layers.0.self_attn.k_proj", "blk.0.attn_k.weight", IN),
              ("model.layers.1.mlp.shared_mlp.gate_proj", "blk.1.ffn_gate_shexp.weight", IN)],
    "hyv4": [("model.layers.0.self_attn.q_a_proj", "blk.0.attn_q_a.weight", IN),
             ("model.layers.1.mlp.shared_experts.down_proj", "blk.1.ffn_down_shexp.weight", IN)],
    "kimi-k3": [("model.layers.0.self_attn.b_proj", "blk.0.ssm_beta.weight", IN),
                ("model.layers.1.mlp.down_proj", "blk.1.ffn_down.weight", IN)],
    "minimax-m3": [("model.layers.0.self_attn.o_proj", "blk.0.attn_output.weight", IN),
                   ("model.layers.1.block_sparse_moe.shared_experts.up_proj", "blk.1.ffn_up_shexp.weight", IN)],
    "muse-glimmer": [("model.layers.0.self_attn.gate_proj", "blk.0.attn_gate.weight", IN),
                     ("model.layers.1.mlp.up_proj", "blk.1.ffn_up.weight", IN)],
    "qwen4exp": [("model.layers.0.linear_attn.in_proj_qkv", "blk.0.attn_qkv.weight", IN),
                 ("model.layers.1.mlp.shared_expert_gate", "blk.1.ffn_gate_inp_shexp.weight", 1)],
}


def test_the_owned_table_covers_every_architecture_gguf_py_cannot_name():
    import gguf

    from gmlx.load.arch_table import ARCH_TABLE

    known = set(gguf.MODEL_ARCH_NAMES.values())
    assert sorted(a for a in ARCH_TABLE if a not in known) == sorted(_OWNED)


@pytest.mark.parametrize("arch", sorted(_OWNED))
def test_an_architecture_gguf_py_cannot_name_exports_through_the_base_names(tmp_path, arch):
    """The pairs are keyed to the base's own tensor names, so the loader
    and llama.cpp both find them, on every architecture the loader reads."""
    import gguf

    targets = _OWNED[arch]
    modules = [(p, *_ab(out, IN, R, seed=i)) for i, (p, _name, out) in enumerate(targets)]
    base_names = [name for _p, name, _out in targets] + ["token_embd.weight", "output_norm.weight"]
    path, n = _save(tmp_path, modules, base_arch=arch, base_names=base_names)
    assert n == 2
    written = sorted(t.name for t in gguf.GGUFReader(path).tensors)
    assert written == sorted(f"{name}.lora_{h}" for _p, name, _o in targets for h in "ab")
    plan = adapter.load_lora_adapter(path, base_arch=arch)
    assert set(plan.modules) == {p for p, _a, _b in modules}
    for module_path, a, b in modules:
        lm = plan.modules[module_path]
        assert lm.transform == "passthrough" and lm.rank == R
        assert mx.array_equal(lm.a, a.astype(mx.float32)) and mx.array_equal(lm.b, b.astype(mx.float32))


def test_a_one_dimensional_gate_exports_and_loads_its_factors_as_they_are(tmp_path):
    """The shared-expert gate of the Qwen MoE families loads from a 1-D
    GGUF tensor through an unsqueeze, which leaves its LoRA factors as
    they are, so both name maps write it and the loader applies it."""
    a, b = _ab(1, IN, R, seed=5)
    path = "model.layers.0.mlp.shared_expert_gate"
    for sub, kw in (("gguf-py", {}), ("base", {"base_names": ["blk.0.ffn_gate_inp_shexp.weight"]})):
        (tmp_path / sub).mkdir()
        out, n = _save(tmp_path / sub, [(path, a, b)], base_arch="qwen35moe", **kw)
        lm = adapter.load_lora_adapter(out, base_arch="qwen35moe").modules[path]
        assert n == 1 and lm.transform == "passthrough" and lm.rank == R
        assert mx.array_equal(lm.a, a) and mx.array_equal(lm.b, b)


def test_modules_the_base_cannot_key_are_named_before_anything_is_written(tmp_path):
    out = tmp_path / "a.gguf"
    conv_a, conv_b = _ab(IN, IN, R, seed=6)
    modules = _modules() + [("model.layers.0.linear_attn.conv1d", conv_a, conv_b)]
    with pytest.raises(ValueError) as e:
        adapter.save_lora_adapter(
            str(out), modules, alpha=ALPHA, base_arch="qwen35", n_head=N_HEAD, n_head_kv=N_HEAD_KV,
            n_layers=2, base_names=["blk.0.attn_q.weight", "blk.0.ffn_down.weight", "blk.0.ssm_conv1d.weight"])
    msg = str(e.value)
    assert msg.startswith("a GGUF adapter cannot hold 2 of the 4 LoRA modules: ")
    assert "model.layers.0.self_attn.k_proj (no tensor of the base GGUF loads into it)" in msg
    assert ("model.layers.0.linear_attn.conv1d (its base tensor blk.0.ssm_conv1d.weight loads through the "
            "'conv1d_unsqueeze' transform, which a LoRA pair cannot follow)") in msg
    assert not out.exists()


def test_two_base_tensors_loading_into_one_module_are_refused():
    names, refused = adapter.adapter_tensor_names(
        ["model.layers.0.self_attn.q_proj"], base_arch="llama",
        base_names=["blk.0.attn_q.weight", "blk.0.attn_q.weight"])
    assert names == {} and refused == {"model.layers.0.self_attn.q_proj": "2 tensors of the base GGUF load into it"}


def test_base_tensor_names_reads_every_shard(tmp_path):
    import gguf
    import numpy as np

    for i, names in enumerate((["blk.0.attn_q.weight"], ["blk.1.attn_q.weight", "output.weight"])):
        w = gguf.GGUFWriter(str(tmp_path / f"m-{i + 1:05d}-of-00002.gguf"), "llama")
        for name in names:
            w.add_tensor(name, np.zeros((2, 2), dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    assert adapter.base_tensor_names(str(tmp_path / "m-00001-of-00002.gguf")) == [
        "blk.0.attn_q.weight", "blk.1.attn_q.weight", "output.weight"]
