#!/usr/bin/env python3
"""MoE routes through the teacher pass and eval on real mlx-lm gates: a
gate route replay cannot recompute weights for is refused at record time
and skipped at replay, and a gate with a weights adapter records and
replays end to end."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

import gmlx.distill as dl

from .test_distill_lib import _BL_MERGES, _bytelevel_tokenizer, _text_corpus, _tiny_cache

# Two MLA layers, both MoE, eight routed experts, two per token, one shared.
_MOE = dict(hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
            routed_scaling_factor=1.0, kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8,
            qk_nope_head_dim=8, v_head_dim=8, n_group=1, topk_group=1, num_experts_per_tok=2,
            moe_layer_freq=1, first_k_dense_replace=0, max_position_embeddings=512, rope_theta=10000.0)
# deepseek_v2's softmax gate has no weights adapter, glm4_moe_lite's sigmoid gate has one
_ARCH = {
    "deepseek_v2": dict(topk_method="greedy", rope_scaling={"factor": 1.0}),
    "glm4_moe_lite": dict(topk_method="noaux_tc"),
}
_ROUTING = {"moe_layers": [0, 1], "k": 2, "n_experts": 8, "dtype": "uint8"}


@pytest.fixture(scope="module")
def tok_bl():
    return _bytelevel_tokenizer(_BL_MERGES)


def _config(arch: str, V: int) -> dict:
    return dict(model_type=arch, vocab_size=V, **_MOE, **_ARCH[arch])


def _tiny_model(arch: str, V: int, seed: int = 0):
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(seed)
    model = mod.Model(mod.ModelArgs.from_dict(_config(arch, V)))
    # the gates start at zeros, which ties every score
    for layer in model.layers:
        layer.mlp.gate.weight = mx.random.normal(layer.mlp.gate.weight.shape)
    mx.eval(model.parameters())
    return model


def _tiny_checkpoint(tmp: Path, tok, arch: str) -> Path:
    from mlx.utils import tree_flatten

    tmp.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tmp)
    cfg = _config(arch, len(dl.token_bytes(tok)))
    model = _tiny_model(arch, cfg["vocab_size"])
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp


def _cache_opts(teacher: Path, tmp: Path):
    from gmlx.distill import teacher as _teacher

    return _teacher.CacheOptions(teacher=str(teacher), corpus=str(_text_corpus(tmp / "c.jsonl")),
                                 out=str(tmp / "cache"), top_k=8, max_len=64, rows_per_shard=2,
                                 routes=True, no_wired_limit=True)


def _kld_arm(model, tok, cache: Path, tmp: Path) -> dict:
    from gmlx.distill import evaluate as _ev

    opts = _ev.EvalOptions(student="s", md=str(tmp / "r.md"), json=str(tmp / "r.json"), kld_cache=str(cache))
    return _ev.run_arm(model, tok, opts, {}, {})["kld"]


def test_route_recording_refuses_a_gate_without_a_weights_adapter():
    """Recording on a gate without a weights adapter is refused, naming the
    layers and the gate, and the recorder comes off; replay_layers_for
    gives no replay on such a model and keeps it on an adapted one."""
    from gmlx.stream.moe_experts import expert_controls_active

    model = _tiny_model("deepseek_v2", 64)
    rec, why = dl.install_route_recording(model)
    assert rec is None
    assert why == ("route replay unsupported on MoE layers [0, 1] "
                   "(no weights adapter for mlx_lm.models.deepseek_v2.MoEGate)")
    assert not any(expert_controls_active(m) for m in model.modules())
    manifest = {"gmlx_distill": {"routing": dict(_ROUTING)}}
    assert dl.replay_layers_for(model, manifest) is None
    assert dl.replay_layers_for(_tiny_model("glm4_moe_lite", 64), manifest) == [0, 1]


def test_cache_routes_refuses_a_teacher_whose_gates_cannot_replay(tmp_path, tok_bl, capsys):
    """The teacher pass refuses --routes before it writes a shard, with the
    reason on the refusal line, instead of minting routes eval cannot use."""
    from gmlx.distill import teacher as _teacher

    teacher = _tiny_checkpoint(tmp_path / "teacher", tok_bl, "deepseek_v2")
    assert _teacher.run_cache(_cache_opts(teacher, tmp_path)) == 2
    err = capsys.readouterr().err
    assert ("[cache] refuse: --routes: route replay unsupported on MoE layers [0, 1] "
            "(no weights adapter for mlx_lm.models.deepseek_v2.MoEGate)") in err
    assert not (tmp_path / "cache" / "manifest.json").exists()


def test_eval_scores_a_routes_cache_without_replay_on_a_gate_without_an_adapter(tmp_path, tok_bl, capsys):
    """A routes cache whose layers match a model with an adapterless gate
    (one minted before the refusal) scores every row without replay
    instead of raising on the first one, and says so."""
    _tiny_cache(tmp_path / "c", tok_bl, routing=dict(_ROUTING))
    model = _tiny_model("deepseek_v2", len(dl.token_bytes(tok_bl)))
    k = _kld_arm(model, tok_bl, tmp_path / "c", tmp_path)
    assert k["rows"] == 6 and k["replayed_rows"] == 0 and k["mean_kld_nats"] is not None
    assert "recorded routes are not replayed" in capsys.readouterr().err


def test_routes_on_an_adapted_gate_record_and_replay_end_to_end(tmp_path, tok_bl):
    """glm4_moe_lite records its routes into the cache and eval replays
    them on every row, the teacher's own weights scoring near zero."""
    from gmlx.distill import student as _student
    from gmlx.distill import teacher as _teacher

    teacher = _tiny_checkpoint(tmp_path / "teacher", tok_bl, "glm4_moe_lite")
    assert _teacher.run_cache(_cache_opts(teacher, tmp_path)) == 0
    man = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    routing = man["gmlx_distill"]["routing"]
    assert routing["moe_layers"] == [0, 1] and routing["k"] == 2 and routing["n_experts"] == 8
    model, _cfg, _tok = _student.load_mlx_student(str(teacher))
    k = _kld_arm(model, tok_bl, tmp_path / "cache", tmp_path)
    assert k["rows"] > 0 and k["replayed_rows"] == k["rows"]
    assert k["mean_kld_nats"] < 1e-3


def test_cache_routes_checks_free_space_once_the_route_bytes_are_known(tmp_path, tok_bl, monkeypatch, capsys):
    """The first space check holds no route bytes, since k is learned from
    the first chunk. The estimate with routes is checked against the free
    space once k is known, and a resume that knows k counts the route bytes
    before it loads the teacher."""
    from gmlx.distill import teacher as _teacher

    teacher = _tiny_checkpoint(tmp_path / "teacher", tok_bl, "glm4_moe_lite")
    opts = _cache_opts(teacher, tmp_path)
    assert _teacher.run_cache(opts) == 0
    out = Path(opts.out)
    prog = json.loads((out / "progress.json").read_text())
    # two MoE layers, two routes each, uint8 ids
    per_pos = 2 * 2 * 1
    plain = dl.estimate_cache_bytes(prog["tokens"], 8, False)
    routed = dl.estimate_cache_bytes(prog["tokens"], 8, False, routes_bytes=per_pos)
    capsys.readouterr()
    # room for the estimate without routes, not with them
    free = (plain + routed) / 2 / 0.9
    monkeypatch.setattr(_teacher._format, "free_bytes", lambda p: int(free))
    fresh = tmp_path / "fresh"
    assert _teacher.run_cache(_teacher.CacheOptions(**dict(vars(opts), out=str(fresh)))) == 2
    err = capsys.readouterr().err
    assert "[cache] routes: k=2" in err and "[cache] refuse: estimate" in err
    assert not list(fresh.glob("batch-*"))
    # a resume with one shard left: the route bytes count before the load
    (out / "batch-00001.safetensors").unlink()
    (out / "manifest.json").unlink()
    done = prog["shards"][0]["bytes"]
    free = ((plain - done) + (routed - done)) / 2 / 0.9
    monkeypatch.setattr(_teacher._format, "free_bytes", lambda p: int(free))
    assert _teacher.run_cache(_teacher.CacheOptions(**dict(vars(opts), resume=True))) == 2
    err = capsys.readouterr().err
    assert "[cache] refuse: estimate" in err and "[cache] routes:" not in err
    assert _teacher._format.route_bytes_per_position(_ROUTING) == per_pos
