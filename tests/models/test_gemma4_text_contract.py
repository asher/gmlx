"""The owned Gemma 4 text model answers what mlx-lm's gemma4_text answers.

Every gmlx text build of gemma4_text resolves ``gmlx.models.gemma4.text``.
Each case builds that class and mlx-lm's ``Model`` from the same config
dict and compares what gmlx reads: the arguments, the attributes, the call
parameters, the parameter tree, ``sanitize`` and the probes that read the
config. Two differences are named: ``sanitize`` drops the k and v weights
of KV-shared layers, and ``score_heads`` reads the real head count where
mlx-lm's class, which has no config, gives the fallback of 32. The cases
are tiny random configs, whose logits are also compared on the CPU
stream, and the Hugging Face configs of google/gemma-4-31B-it,
gemma-4-26B-A4B-it, gemma-4-12b-it, gemma-4-E4B-it and gemma-4-E2B-it,
flattened to their text config as an hf_source load flattens them.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from gmlx._vendor.mlx_lm.models import gemma4_text as g4t

from gmlx.models.gemma4 import language as owned_lang
from gmlx.models.gemma4 import text

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_BASE = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
    vocab_size_per_layer_input=128, num_key_value_heads=2,
    num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    sliding_window=8, sliding_window_pattern=3, tie_word_embeddings=True,
)
TINY = {
    "dense": {},
    "ple": {"num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8},
    "moe": {"enable_moe_block": True, "num_experts": 4, "top_k_experts": 2,
            "moe_intermediate_size": 32, "attention_k_eq_v": True,
            "num_global_key_value_heads": 1},
    "hf_rope": {"rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.5,
                           "rope_theta": 5e5, "rope_type": "proportional"},
        "sliding_attention": {"partial_rotary_factor": 1.0,
                              "rope_theta": 1e4, "rope_type": "default"}},
        "use_bidirectional_attention": "vision"},
}
HF = {
    "31b_hf": "gemma4_31b_it_config.json",
    "26b_a4b_hf": "gemma4_26b_a4b_it_config.json",
    "12b_hf": "gemma4_12b_it_config.json",
    "e4b_hf": "gemma4_e4b_it_config.json",
    "e2b_hf": "gemma4_e2b_it_config.json",
}
PROMPT = [3, 17, 42, 99, 7, 63, 5, 28, 11, 90, 4, 8, 9, 10]


def _config(name) -> dict:
    if name in HF:
        from gmlx.load.loader import _text_config_from_container

        return _text_config_from_container(json.loads(
            (_FIXTURES / HF[name]).read_text()))
    return {**_BASE, **TINY[name]}


def _pair(cfg):
    """mlx-lm's model and the owned one, built lazily from ``cfg``."""
    ref = g4t.Model(g4t.ModelArgs.from_dict(json.loads(json.dumps(cfg))))
    got = text.Model(text.ModelArgs.from_dict(json.loads(json.dumps(cfg))))
    return ref, got


@pytest.fixture
def owned_softcap(monkeypatch):
    """mlx-lm's class with the float32 softcap every gmlx load gives it."""
    monkeypatch.setattr(g4t, "logit_softcap", owned_lang.logit_softcap)


def _tiny(name):
    mx.random.seed(1)
    ref, got = _pair(_config(name))
    mx.eval(ref.parameters())
    got.load_weights(list(tree_flatten(ref.parameters())), strict=True)
    mx.eval(got.parameters())
    return ref, got


def test_the_resolver_owns_gemma4_text_only():
    from gmlx._vendor.mlx_lm.models import llama

    assert text.get_model_classes({"model_type": "gemma4_text"}) == (
        text.Model, text.ModelArgs)
    assert text.get_model_classes({"model_type": "llama"}) == (
        llama.Model, llama.ModelArgs)


def test_the_presence_gate_needs_no_mlx_lm_module(monkeypatch):
    import importlib.util

    from gmlx.load import arch_table

    real = importlib.util.find_spec

    def no_stock(name, *a, **k):
        return None if name == "gmlx._vendor.mlx_lm.models.gemma4_text" else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", no_stock)
    assert arch_table.mlx_lm_has_model("gemma4_text")


CASES = [*TINY, *HF]


def test_each_hf_config_flattens_to_gemma4_text():
    for name in HF:
        assert _config(name)["model_type"] == "gemma4_text", name


@pytest.mark.parametrize("name", CASES)
def test_args_carry_every_mlx_lm_field(name):
    ref, got = _pair(_config(name))
    assert got.args is got.config
    for field in dataclasses.fields(g4t.ModelArgs):
        assert getattr(got.args, field.name) == getattr(ref.args, field.name), (
            field.name)


@pytest.mark.parametrize("name", CASES)
def test_the_bridge_leaves_out_image_attention(name):
    from gmlx.gen.media_spans import vision_blocks

    ref, got = _pair(_config(name))
    assert got.args.use_bidirectional_attention is None
    assert vision_blocks(got) is vision_blocks(ref) is False


@pytest.mark.parametrize("name", CASES)
def test_attributes_match(name):
    ref, got = _pair(_config(name))
    for attr in ("model_type", "tie_word_embeddings", "final_logit_softcapping",
                 "head_dim", "n_kv_heads"):
        assert getattr(got, attr) == getattr(ref, attr), attr
    assert not hasattr(got, "lm_head") and not hasattr(ref, "lm_head")
    assert got.model.embed_scale == ref.model.embed_scale
    assert len(got.layers) == len(ref.layers)
    assert [(type(c).__name__, getattr(c, "max_size", None), getattr(c, "keep", None))
            for c in got.make_cache()] == [
        (type(c).__name__, getattr(c, "max_size", None), getattr(c, "keep", None))
        for c in ref.make_cache()]
    assert (list(inspect.signature(got.__call__).parameters)
            == list(inspect.signature(ref.__call__).parameters))
    assert ({k: v.shape for k, v in tree_flatten(got.parameters())}
            == {k: v.shape for k, v in tree_flatten(ref.parameters())})


def _close(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        mx.allclose(a, b, rtol=0, atol=1e-5).item())


@pytest.mark.parametrize("name", ["dense", "ple", "moe", "hf_rope"])
def test_logits_match_mlx_lm(name, owned_softcap):
    """A prefill past the window, greedy decode steps, and the embedding
    calls TextOnlyModel makes."""
    with mx.stream(mx.cpu):
        ref, got = _tiny(name)
        ids = mx.array([PROMPT])
        rc, gc = ref.make_cache(), got.make_cache()
        a, b = ref(ids, cache=rc), got(ids, cache=gc)
        assert _close(a, b)
        for _ in range(3):
            step = mx.argmax(a[:, -1:], axis=-1)
            a, b = ref(step, cache=rc), got(step, cache=gc)
            assert _close(a, b)
        e = ref.model.embed_tokens(ids)
        assert _close(ref(ids, input_embeddings=e), got(ids, input_embeddings=e))
        if ref.model.hidden_size_per_layer_input:
            pli = ref.model._get_per_layer_inputs(ids)
            assert _close(ref(None, input_embeddings=e, per_layer_inputs=pli),
                          got(None, input_embeddings=e, per_layer_inputs=pli))
        else:
            assert _close(ref(None, input_embeddings=e),
                          got(None, input_embeddings=e))


def test_embeddings_without_ids_on_a_ple_model_raise():
    _, got = _pair(_config("ple"))
    e = mx.zeros((1, 3, _BASE["hidden_size"]))
    with pytest.raises(ValueError, match="per_layer_inputs"):
        got(None, input_embeddings=e)


def test_an_untied_config_is_refused_by_name():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        text.Model(text.ModelArgs.from_dict(
            {**_BASE, "tie_word_embeddings": False}))


def _shared_kv(name, first) -> bool:
    parts = name.split(".")
    return (parts[:2] == ["model", "layers"] and int(parts[2]) >= first
            and parts[3] == "self_attn"
            and parts[4] in {"k_proj", "v_proj", "k_norm", "v_norm"})


@pytest.mark.parametrize("name", ["dense", "ple", "moe"])
def test_sanitize_matches_mlx_lm_apart_from_the_kv_shared_drop(name):
    """On the remap names, and on HF checkpoint names: rotary tables, clip
    ranges, fused expert projections and the k and v weights of
    KV-shared layers."""
    ref, got = _pair(_config(name))
    weights = {k: mx.zeros((2,)) for k, _ in tree_flatten(ref.parameters())}
    weights.update({
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((2,)),
        "model.layers.0.mlp.gate_proj.input_max": mx.zeros((1,)),
        "model.layers.0.experts.gate_up_proj": mx.zeros((4, 64, 16)),
        "model.layers.0.experts.down_proj": mx.zeros((4, 16, 32)),
    })
    for layer in (1, 4, 5):
        for leaf in ("k_proj.weight", "v_proj.weight", "k_norm.weight",
                     "v_norm.weight"):
            weights[f"model.layers.{layer}.self_attn.{leaf}"] = mx.zeros((2,))
    want = ref.sanitize(dict(weights))
    have = got.sanitize(dict(weights))
    first = _BASE["num_hidden_layers"] - got.args.num_kv_shared_layers
    dropped = {k for k in want
               if got.args.num_kv_shared_layers and _shared_kv(k, first)}
    assert set(have) == set(want) - dropped
    assert bool(dropped) == bool(got.args.num_kv_shared_layers)
    assert all(have[k] is want[k] or mx.array_equal(have[k], want[k]).item()
               for k in have)


@pytest.mark.parametrize("name", CASES)
def test_the_config_probes_answer_as_on_mlx_lm(name):
    from gmlx.cache.kvarn_cache import kvarn_head_dims
    from gmlx.distill.head import head_spec_from_model, model_args
    from gmlx.gen.diffusion import is_diffusion_model
    from gmlx.gen.prefill_decay import resolve_score_profile, score_heads
    from gmlx.serve.patches.sampling import native_context

    ref, got = _pair(_config(name))
    assert native_context(got) == native_context(ref)
    assert resolve_score_profile(got, None) == resolve_score_profile(ref, None)
    assert is_diffusion_model(got) is is_diffusion_model(ref) is False
    assert kvarn_head_dims(got) == kvarn_head_dims(ref)
    assert model_args(got) is got.args
    rs, gs = head_spec_from_model(ref), head_spec_from_model(got)
    assert (gs.softcap, gs.scale) == (rs.softcap, rs.scale)
    # The named difference: mlx-lm's class has no config.
    assert score_heads(ref) == 32
    assert score_heads(got) == got.args.num_attention_heads


def test_training_attention_patches_the_owned_language_module():
    import gmlx.tune.attention as tune_attention
    from gmlx.models.gemma4 import dispatch

    _, got = _pair(_config("dense"))
    restore = tune_attention.install_training_attention(got)
    try:
        assert (owned_lang.scaled_dot_product_attention
                is not dispatch.scaled_dot_product_attention)
        assert not hasattr(text, "scaled_dot_product_attention")
    finally:
        restore()
    assert (owned_lang.scaled_dot_product_attention
            is dispatch.scaled_dot_product_attention)


# Tier 2: the config each Gemma 4 GGUF synthesizes.


def test_each_gemma4_gguf_config_builds_the_contract(gguf_index):
    from gmlx.commands.tool_preflight import _synth_config
    from gmlx.gen.prefill_decay import score_heads

    paths = gguf_index.get("gemma4", [])
    if not paths:
        pytest.skip("no Gemma 4 GGUF under KQUANT_TEST_GGUF_DIR")
    for path in paths:
        cfg = _synth_config(path)
        if cfg is None or cfg.get("model_type") != "gemma4_text":
            continue
        ref, got = _pair(cfg)
        for field in dataclasses.fields(g4t.ModelArgs):
            assert getattr(got.args, field.name) == getattr(
                ref.args, field.name), (path, field.name)
        assert ({k: v.shape for k, v in tree_flatten(got.parameters())}
                == {k: v.shape for k, v in tree_flatten(ref.parameters())}), path
        assert score_heads(got) == cfg["num_attention_heads"], path
