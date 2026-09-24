#!/usr/bin/env python3
"""LoRA train -> GGUF save fidelity (train T2): the extraction + orientation
transpose + q/k forward-permute + scale chain must reconstruct an mlx-lm
``LoRALinear``'s exact delta after a save -> ``load_lora_adapter`` round-trip.
CPU-only - hand-built LoRALinear layers, no training, no kernels."""
from __future__ import annotations

import os

import mlx.core as mx
import pytest

pytest.importorskip("gguf")
import mlx.nn as nn  # noqa: E402

import gmlx.load.adapter as adapter  # noqa: E402
import gmlx.commands.train as train  # noqa: E402
from gmlx.load.transforms import qk_permute_wire  # noqa: E402

LoRALinear = pytest.importorskip("mlx_lm.tuner.lora").LoRALinear

N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 8
IN, R, S = 16, 4, 2.0
Q_OUT, K_OUT = N_HEAD * HEAD_DIM, N_HEAD_KV * HEAD_DIM   # 32, 16
CONFIG = {"num_attention_heads": N_HEAD, "num_key_value_heads": N_HEAD_KV,
          "num_hidden_layers": 1}


# The base tensors _Model's three LoRA modules load from.
_BASE_NAMES = ("blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.ffn_down.weight")


def _lora(out, *, seed):
    mx.random.seed(seed)
    ll = LoRALinear(input_dims=IN, output_dims=out, r=R, scale=S)
    ll.lora_a = mx.random.normal((IN, R))   # lora_b inits to zero -> a real delta
    ll.lora_b = mx.random.normal((R, out))
    return ll


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _lora(Q_OUT, seed=1)
        self.k_proj = _lora(K_OUT, seed=2)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = _lora(IN, seed=3)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer()]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()


def _delta(ll, x):
    """The LoRALinear's adapter contribution alone (forward minus the base)."""
    return ll(x) - ll.linear(x)


def test_trained_lora_roundtrips_to_gguf_delta(tmp_path):
    model = _Model()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)

    out = str(tmp_path / "trained.gguf")
    n = train.save_trained_adapter(model, CONFIG, base_arch="llama",
                                   out_path=out, rank=R, scale=S)
    assert n == 3

    plan = adapter.load_lora_adapter(out)
    assert plan.alpha == pytest.approx(S * R)
    assert set(plan.modules) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.down_proj",
    }

    layer = model.model.layers[0]
    cases = {
        "model.layers.0.self_attn.q_proj": (layer.self_attn.q_proj, N_HEAD),
        "model.layers.0.self_attn.k_proj": (layer.self_attn.k_proj, N_HEAD_KV),
        "model.layers.0.mlp.down_proj": (layer.mlp.down_proj, None),
    }
    x = mx.random.normal((3, IN))
    for path, (ll, nh) in cases.items():
        lm = plan.modules[path]
        assert lm.scale == pytest.approx(S)           # alpha/rank == trained scale
        b = qk_permute_wire(lm.b, nh) if lm.transform == "qk_permute" else lm.b
        recon = lm.scale * (x @ lm.a.T) @ b.T         # the loader+install forward
        assert mx.allclose(recon, _delta(ll, x), atol=1e-5)


def test_no_lora_layers_raises(tmp_path):
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(IN, IN)

    with pytest.raises(ValueError, match="no trained LoRA layers"):
        train.save_trained_adapter(Plain(), CONFIG, base_arch="llama",
                                   out_path=str(tmp_path / "x.gguf"),
                                   rank=R, scale=S)


def test_grad_checkpoint_uses_the_layer_class_checkpointer(monkeypatch, tmp_path):
    """--grad-checkpoint wraps every decoder-layer class through
    gmlx.tune.checkpoint, which also enters a language_model stack; mlx-lm's
    own grad_checkpoint (layer 0's class only) stays off."""
    import contextlib

    import mlx_lm.tuner.datasets as datasets
    import mlx_lm.tuner.trainer as trainer
    import mlx_kquant.mlx_lm_patch as patch

    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight
    import gmlx.tune.attention as attention
    import gmlx.tune.checkpoint as checkpoint
    import gmlx.tune.gdn as gdn

    seen = {}
    model = _Model()
    monkeypatch.setattr(patch, "patch_mlx_lm_lora", lambda: None)
    monkeypatch.setattr(preflight, "preflight", lambda p, hf_source=None: type("P", (), {"arch": "llama"})())
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: list(_BASE_NAMES))
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", lambda p, hf_source=None: (model, CONFIG, object()))
    monkeypatch.setattr(train, "prepare_lora_student", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(attention, "install_training_attention", lambda m: (lambda: None))
    monkeypatch.setattr(gdn, "install_training_gdn", lambda m: None)
    monkeypatch.setattr(checkpoint, "checkpoint_layers", lambda m: seen.setdefault("ckpt", m) and 2)
    monkeypatch.setattr(trainer, "train", lambda m, o, tr, va, args: seen.setdefault("args", args))
    monkeypatch.setattr(train, "save_trained_adapter", lambda *a, **k: 0)
    train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"),
                     iters=1, grad_checkpoint=True)
    assert seen["ckpt"] is model
    assert seen["args"].grad_checkpoint is False


def test_cmd_train_refuses_grad_checkpoint_with_dropout(tmp_path, capsys):
    """The compiled train step cannot replay a layer's dropout mask in the
    backward recompute, so the pair is refused before any model load."""
    rc = train.cmd_train(["base.gguf", "--data", str(tmp_path), "--adapter-out", str(tmp_path / "o.gguf"),
                          "--grad-checkpoint", "--dropout", "0.1"])
    assert rc == 2
    assert "--grad-checkpoint recomputes each layer under a fresh dropout mask" in capsys.readouterr().err


def test_adapter_alpha_follows_the_trained_rank(tmp_path):
    """The stored alpha is the multiplier times the factors' own rank; a
    rank argument that disagrees with the factors is refused rather than
    written."""
    model = _Model()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)
    out = str(tmp_path / "trained.gguf")
    assert train.save_trained_adapter(model, CONFIG, base_arch="llama", out_path=out, scale=S) == 3
    assert adapter.load_lora_adapter(out).alpha == pytest.approx(S * R)
    with pytest.raises(ValueError, match="rank"):
        train.save_trained_adapter(model, CONFIG, base_arch="llama", out_path=out, rank=R + 1, scale=S)


def test_adapter_export_is_atomic(tmp_path, monkeypatch):
    """A failure while the adapter is written leaves no file at the
    output path, and a good write leaves no temporary beside it."""
    import gmlx.tune.lora as lora

    model = _Model()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)
    out = tmp_path / "trained.gguf"
    real = lora.save_lora_adapter

    def partial(path, *a, **k):
        with open(path, "wb") as fh:
            fh.write(b"GGUF" + bytes(16))
        raise RuntimeError("disk full")

    monkeypatch.setattr(lora, "save_lora_adapter", partial)
    with pytest.raises(RuntimeError, match="disk full"):
        train.save_trained_adapter(model, CONFIG, base_arch="llama", out_path=str(out), scale=S)
    assert not out.exists()
    monkeypatch.setattr(lora, "save_lora_adapter", real)
    assert train.save_trained_adapter(model, CONFIG, base_arch="llama", out_path=str(out), scale=S) == 3
    assert out.exists() and [p.name for p in tmp_path.iterdir()] == ["trained.gguf"]


def test_probe_writable_refuses_a_directory(tmp_path):
    """A path that names an existing directory cannot take the adapter
    file, and the probe says so rather than passing it to the export."""
    assert train.probe_writable(str(tmp_path / "new" / "a.gguf")) is None
    err = train.probe_writable(str(tmp_path))
    assert err and "directory" in err


def test_lora_rank_below_one_and_a_slash_terminated_adapter_path_are_refused(tmp_path):
    """Rank 0 would write factors no loader can scale; a path ending in a
    separator names a directory the export could not replace."""
    from gmlx.tune.lora import lora_scale

    with pytest.raises(ValueError, match="rank"):
        lora_scale(0, 2.0)
    with pytest.raises(ValueError, match="rank"):
        lora_scale(-1, None, 4.0)
    err = train.probe_writable(str(tmp_path / "new") + "/")
    assert err and "directory" in err


def test_lora_modules_to_gguf_refuses_a_factor_that_is_not_a_matrix(tmp_path):
    """A stacked expert factor (three axes) cannot be transposed into the
    two-axis tensors a GGUF adapter holds; the export names the module
    rather than writing a wrong tensor."""
    from gmlx.tune.lora import lora_modules_to_gguf

    class _Stacked(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.zeros((2, 3, 4))
            self.lora_b = mx.zeros((2, 4, 3))

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Stacked()]

    with pytest.raises(ValueError, match="layers.0"):
        lora_modules_to_gguf(_M())


def test_train_rank_below_one_is_refused_at_parse_time(tmp_path, capsys):
    """The rank reaches the factors' shapes; zero or less is refused by the
    parser, before the base model is resolved."""
    with pytest.raises(SystemExit) as e:
        train.cmd_train([str(tmp_path / "m.gguf"), "--data", str(tmp_path), "--adapter-out",
                         str(tmp_path / "a.gguf"), "--rank", "0"])
    assert e.value.code == 2 and "at least 1" in capsys.readouterr().err


@pytest.mark.parametrize("dropout", ["1", "-0.1", "nan"])
def test_train_dropout_outside_zero_to_one_is_refused_at_parse_time(tmp_path, capsys, dropout):
    """nn.Dropout refuses a probability of 1 or more inside the adapter
    install, after the base model load; the parser refuses it first."""
    with pytest.raises(SystemExit) as e:
        train.cmd_train([str(tmp_path / "m.gguf"), "--data", str(tmp_path), "--adapter-out",
                         str(tmp_path / "a.gguf"), "--dropout", dropout])
    assert e.value.code == 2 and "below 1" in capsys.readouterr().err


def test_adapter_export_reads_the_text_model_under_a_multimodal_wrapper(tmp_path):
    """A wrapper that nests the text model under language_model (the
    mlx-vlm layout) exports the same module paths as the text model
    alone, so the adapter loads onto its base."""
    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _Model()

    model = _Wrapper()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)
    out = str(tmp_path / "wrapped.gguf")
    assert train.save_trained_adapter(model, CONFIG, base_arch="llama", out_path=out, rank=R, scale=S) == 3
    plan = adapter.load_lora_adapter(out)
    assert set(plan.modules) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.down_proj",
    }


def test_grad_checkpoint_refusal_restores_attention_and_exits_2(monkeypatch, tmp_path, capsys):
    """A layer class that refuses per-layer checkpointing ends train_lora
    before the train loop with the training attention taken off again, and
    the command reports it and exits 2."""
    import contextlib

    import mlx_lm.tuner.datasets as datasets
    import mlx_lm.tuner.trainer as trainer
    import mlx_kquant.mlx_lm_patch as patch

    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight
    import gmlx.tune.attention as attention
    import gmlx.tune.checkpoint as checkpoint
    import gmlx.tune.gdn as gdn

    seen = []
    model = _Model()

    def refuse(m):
        raise ValueError("per-layer checkpointing cannot run _Layer: its layers share a bank")

    monkeypatch.setattr(patch, "patch_mlx_lm_lora", lambda: None)
    monkeypatch.setattr(preflight, "preflight", lambda p, hf_source=None: type("P", (), {"arch": "llama"})())
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: list(_BASE_NAMES))
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", lambda p, hf_source=None: (model, CONFIG, object()))
    monkeypatch.setattr(train, "prepare_lora_student", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(attention, "install_training_attention", lambda m: (lambda: seen.append("restored")))
    monkeypatch.setattr(gdn, "install_training_gdn", lambda m: None)
    monkeypatch.setattr(checkpoint, "checkpoint_layers", refuse)
    monkeypatch.setattr(trainer, "train", lambda *a, **k: seen.append("trained"))
    orig = mx.argpartition
    with pytest.raises(train.TrainRefused, match="--grad-checkpoint: per-layer checkpointing cannot run _Layer"):
        train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"), iters=1, grad_checkpoint=True)
    assert seen == ["restored"]
    assert mx.argpartition is orig

    def refused(*a, **k):
        raise train.TrainRefused("--grad-checkpoint: per-layer checkpointing cannot run _Layer")

    monkeypatch.setattr(train, "train_lora", refused)
    base = tmp_path / "base.gguf"
    base.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text": "a b c"}\n')
    rc = train.cmd_train([str(base), "--data", str(data), "--adapter-out", str(tmp_path / "o.gguf"),
                          "--grad-checkpoint"])
    assert rc == 2
    assert "error: --grad-checkpoint: per-layer checkpointing cannot run _Layer" in capsys.readouterr().err


def test_train_loop_runs_with_selection_ids_off_the_gradient(monkeypatch, tmp_path):
    """mlx-lm's train loop runs with argpartition and its siblings detached,
    so a MoE base's router has a backward, and train_lora puts the
    originals back when the loop ends."""
    import contextlib

    import mlx_lm.tuner.datasets as datasets
    import mlx_lm.tuner.trainer as trainer
    import mlx_kquant.mlx_lm_patch as patch

    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight
    import gmlx.tune.attention as attention
    import gmlx.tune.gdn as gdn

    seen = []
    model = _Model()
    orig = mx.argpartition
    monkeypatch.setattr(patch, "patch_mlx_lm_lora", lambda: None)
    monkeypatch.setattr(preflight, "preflight", lambda p, hf_source=None: type("P", (), {"arch": "llama"})())
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: list(_BASE_NAMES))
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", lambda p, hf_source=None: (model, CONFIG, object()))
    monkeypatch.setattr(train, "prepare_lora_student", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(attention, "install_training_attention", lambda m: (lambda: None))
    monkeypatch.setattr(gdn, "install_training_gdn", lambda m: None)
    monkeypatch.setattr(trainer, "train",
                        lambda *a, **k: seen.append((getattr(mx.argpartition, "_gmlx_index_stop_gradient", False),
                                                     os.environ.get("KQ_SWITCH_GEMM_MIN_ROWS"))))
    monkeypatch.setattr(train, "save_trained_adapter", lambda *a, **k: 0)
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "512")
    train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"), iters=1)
    # kq's segment GEMM for large sorted expert calls has no backward
    assert seen == [(True, "0")]
    assert mx.argpartition is orig
    assert os.environ["KQ_SWITCH_GEMM_MIN_ROWS"] == "512"


def test_a_train_loop_that_raises_restores_the_selection_ops_and_attention(monkeypatch, tmp_path):
    import contextlib

    import mlx_lm.tuner.datasets as datasets
    import mlx_lm.tuner.trainer as trainer
    import mlx_kquant.mlx_lm_patch as patch

    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight
    import gmlx.tune.attention as attention
    import gmlx.tune.gdn as gdn

    seen = []
    orig = mx.argpartition

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(patch, "patch_mlx_lm_lora", lambda: None)
    monkeypatch.setattr(preflight, "preflight", lambda p, hf_source=None: type("P", (), {"arch": "llama"})())
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: list(_BASE_NAMES))
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", lambda p, hf_source=None: (_Model(), CONFIG, object()))
    monkeypatch.setattr(train, "prepare_lora_student", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(attention, "install_training_attention", lambda m: (lambda: seen.append("restored")))
    monkeypatch.setattr(gdn, "install_training_gdn", lambda m: None)
    monkeypatch.setattr(trainer, "train", boom)
    monkeypatch.delenv("KQ_SWITCH_GEMM_MIN_ROWS", raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"), iters=1)
    assert mx.argpartition is orig and seen == ["restored"]
    assert "KQ_SWITCH_GEMM_MIN_ROWS" not in os.environ


def test_train_refuses_before_the_loop_when_the_adapter_could_not_hold_a_module(monkeypatch, tmp_path):
    """A LoRA module that no tensor of the base loads into ends train_lora
    before the first step, and a run that trains hands the base names to
    the export."""
    import contextlib

    import mlx_lm.tuner.datasets as datasets
    import mlx_lm.tuner.trainer as trainer
    import mlx_kquant.mlx_lm_patch as patch

    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight
    import gmlx.tune.attention as attention
    import gmlx.tune.gdn as gdn

    seen = []
    monkeypatch.setattr(patch, "patch_mlx_lm_lora", lambda: None)
    monkeypatch.setattr(preflight, "preflight", lambda p, hf_source=None: type("P", (), {"arch": "llama"})())
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", lambda p, hf_source=None: (_Model(), CONFIG, object()))
    monkeypatch.setattr(train, "prepare_lora_student", lambda *a, **k: None)
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(attention, "install_training_attention", lambda m: (lambda: None))
    monkeypatch.setattr(gdn, "install_training_gdn", lambda m: None)
    monkeypatch.setattr(trainer, "train", lambda *a, **k: seen.append("trained"))
    monkeypatch.setattr(train, "save_trained_adapter", lambda *a, **k: seen.append(k["base_names"]) or 3)
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: ["blk.0.attn_q.weight", "blk.0.ffn_down.weight"])
    with pytest.raises(train.TrainRefused, match=r"^a GGUF adapter cannot hold 1 of the adapted modules, nothing "
                                                 r"was trained: model.layers.0.self_attn.k_proj \(no tensor"):
        train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"), iters=1)
    assert seen == []
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: list(_BASE_NAMES))
    assert train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"), iters=1)[1] == 3
    assert seen == ["trained", list(_BASE_NAMES)]
