"""Inkling 2 loading: both GGUF layouts, the config synth's refusals, the
fused expert scale on shared loop passes, and the float32 pins."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
import inkling2_tiny as T  # noqa: E402

from gmlx.load.config_synth import synthesize_config  # noqa: E402

IDS = [[5, 9, 30, 41, 7, 66, 12, 3, 90, 100, 15, 22, 61, 80, 33, 44, 55, 70,
        21, 10]]


def _synth(layout, **edits):
    v = T.values()
    meta = T.metadata(layout, v["token_embd.weight"].shape[0])
    arch = meta["general.architecture"]
    for key, val in edits.items():
        name = f"{arch}.{key}" if not key.startswith("tokenizer.") else key
        if val is None:
            meta.pop(name, None)
        else:
            meta[name] = val
    return synthesize_config(meta, tensor_shapes=T.shapes(layout, v))


def test_both_layouts_synthesize_the_same_config():
    root, mainline = _synth("root"), _synth("mainline")
    assert root == mainline
    assert root["model_type"] == "inkling2"
    assert root["num_hidden_layers"] == T.LAYERS
    assert root["layer_visits"] == T.visits()
    assert root["loop"] == {"start": T.START, "end": T.END,
                            "passes": T.PASSES, "gate_rank": T.RANK}
    assert root["unpadded_vocab_size"] == len(T.tokens())
    assert root["vocab_size"] > root["unpadded_vocab_size"]


def test_an_inkling2_file_without_loop_keys_runs_one_pass():
    edits = {"num_loops": None, "loop.core_start": None, "loop.core_end": None,
             "block_count": T.LAYERS,
             "attention.head_count": [T.HEADS] * T.LAYERS,
             "attention.head_count_kv": [T.KV] * T.LAYERS,
             "attention.sliding_window_pattern": T.SWA}
    c = _synth("root", **edits)
    assert c["loop"] is None
    assert c["layer_visits"] == list(range(T.LAYERS))


@pytest.mark.parametrize("layout, edits, exc, match", [
    ("mainline", {"loop.varying": True}, NotImplementedError, "loop.varying"),
    ("root", {"block_count": len(T.visits()) - 1}, ValueError, "entries"),
    # The second visit of layer START says full, the first sliding.
    ("root", {"attention.sliding_window_pattern":
              [T.SWA[p] != (v == T.END) for v, p in enumerate(T.visits())]},
     ValueError, "differs between visits"),
    ("mainline", {"loop.end": T.END + 1}, ValueError, "invalid loop core"),
    ("mainline", {"vocab_size_unpadded": len(T.tokens()) - 1}, ValueError,
     "tokenizer size"),
])
def test_synth_refuses_what_the_model_cannot_run(layout, edits, exc, match):
    with pytest.raises(exc, match=match):
        _synth(layout, **edits)


def test_the_model_refuses_a_variable_loop_count():
    from gmlx.models.inkling2.model import ModelArgs

    c = _synth("mainline")
    c["loop"] = dict(c["loop"], varying=True)
    with pytest.raises(NotImplementedError, match="variable loop"):
        ModelArgs.from_dict(c)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Loads a tiny GGUF written from values ``v``; returns its logits."""
    import gmlx

    root = tmp_path_factory.mktemp("inkling2")
    models = {}

    def load(name, layout, v, **kw):
        path = root / f"{name}.gguf"
        T.write(path, layout, v, **kw)
        models[name] = gmlx.load_model(str(path), verbose=False)[0]
        return models[name]

    def logits(model):
        return np.array(model(mx.array(IDS)).astype(mx.float32))

    load.logits = logits
    load.models = models
    return load


def test_both_layouts_give_the_same_logits(tiny):
    v = T.values()
    root = tiny.logits(tiny("root", "root", v))
    mainline = tiny.logits(tiny("mainline", "mainline", v))
    assert root.shape == (1, len(IDS[0]), len(T.tokens()))
    np.testing.assert_array_equal(root, mainline)


def test_the_message_query_rows_reach_the_logits(tiny):
    # A mainline file whose message rows are in another order: the
    # equality above would not hold if the packed split were wrong.
    v = T.values()
    good = tiny.logits(tiny("order_good", "mainline", v))
    names = T._tensor_names

    def swapped(values, layout):
        for name, a, bf16 in names(values, layout):
            if name.endswith("loop_msg_query.weight"):
                a = a[[0, 3, 1, 2]]
            yield name, a, bf16

    T._tensor_names = swapped
    try:
        bad = tiny.logits(tiny("order_bad", "mainline", v))
    finally:
        T._tensor_names = names
    assert np.abs(good - bad).max() > 1e-2


def test_controllers_and_norms_load_float32(tiny):
    from mlx.utils import tree_flatten

    model = tiny.models.get("root") or tiny("root", "root", T.values())
    params = dict(tree_flatten(model.parameters()))
    loop = [k for k in params if ".loop." in k]
    norms = [k for k in params if k.endswith("norm.weight")]
    assert loop and norms
    assert {str(params[k].dtype) for k in loop + norms} == {"mlx.core.float32"}


def test_expert_scale_applies_before_the_activation_on_every_pass(tiny):
    # Power-of-two scales fold into Q8_0 weights exactly. A scale that is
    # dropped, applied after the activation or missing on a later loop
    # pass of a core layer breaks the match with the folded file.
    rng = np.random.default_rng(1)
    scales = {i: rng.choice([0.5, 2.0, 4.0], T.EXPERTS)
              for i in range(T.LAYERS)}
    folded = T.values()
    for i, s in scales.items():
        name = f"blk.{i}.ffn_gate_up_exps.weight"
        folded[name] = folded[name] * s[:, None, None].astype(np.float32)
    scaled = tiny.logits(tiny("scaled", "root", T.values(scales=scales),
                              experts_q8=True))
    want = tiny.logits(tiny("folded", "root", folded, experts_q8=True))
    plain = tiny.logits(tiny("plain", "root", T.values(), experts_q8=True))
    np.testing.assert_allclose(scaled, want, atol=1e-5)
    assert np.abs(scaled - plain).max() > 1e-1


def test_kv_bits_are_dropped_with_a_reason(tiny):
    from gmlx.cache.kv_policy import resolve_kv_quant_policy

    model = tiny.models.get("root") or tiny("root", "root", T.values())
    policy = resolve_kv_quant_policy(model.make_cache(), kv_bits=4)
    assert policy.verdict == "dropped"
    assert policy.reason
