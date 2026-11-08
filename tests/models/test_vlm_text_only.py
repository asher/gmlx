"""The vendored text-only adapter (gmlx.models.vlm_text_only, from mlx-vlm 0.6.4
models/text_only.py). Vendoring moved the module out of mlx_vlm.models, so
package-relative imports the original relied on must have been rewritten;
the make_cache fallback's `.cache` was missed, and every served request
against an mlx-lm model without its own make_cache died in
make_prompt_cache (ModuleNotFoundError: gmlx.cache)."""
import mlx.nn as nn
import pytest

from gmlx.models.vlm_text_only import LanguageModel


class _BareModel(nn.Module):
    """An mlx-lm-style model with no make_cache of its own."""

    layers = [object(), object()]


def test_make_cache_fallback_builds_mlx_vlm_caches():
    from gmlx._vendor.mlx_vlm.models.cache import KVCache

    caches = LanguageModel(_BareModel()).make_cache()
    assert len(caches) == 2
    assert all(isinstance(c, KVCache) for c in caches)


def test_make_cache_prefers_the_models_own():
    class _Own(nn.Module):
        layers = [object()]

        def make_cache(self):
            return ["own"]

    assert LanguageModel(_Own()).make_cache() == ["own"]


def _llama():
    import mlx.core as mx

    from gmlx._vendor.mlx_lm.models import llama

    mx.random.seed(0)
    args = llama.ModelArgs(model_type="llama", hidden_size=32,
                           num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=4, num_key_value_heads=2,
                           rms_norm_eps=1e-5, vocab_size=64)
    lm = LanguageModel(llama.Model(args))
    mx.eval(lm.parameters())
    return lm


def test_logits_to_keep_projects_only_the_last_columns():
    """The trunk runs every column and hands the head the kept ones, from
    token ids or embeddings alike. The trunk is the model's own again after
    the forward."""
    import mlx.core as mx
    import numpy as np

    lm = _llama()
    assert lm.supports_logits_to_keep
    trunk = lm._model.model
    ids = mx.array([[3, 9, 4, 7, 1], [5, 2, 8, 8, 6]])
    full = lm(ids).logits
    assert full.shape == (2, 5, 64)
    for kw in ({}, {"inputs_embeds": lm.input_embeds(ids)}):
        kept = lm(ids, logits_to_keep=2, **kw).logits
        assert kept.shape == (2, 2, 64)
        np.testing.assert_allclose(np.array(kept), np.array(full[:, -2:]),
                                   atol=1e-4)
        assert "model" not in vars(lm._model)
        assert lm._model.model is trunk
    assert lm(ids, logits_to_keep=0).logits.shape == (2, 5, 64)


def test_kept_columns_slice_only_on_the_arming_thread():
    import threading

    import mlx.core as mx

    from gmlx.models.vlm_text_only import _KeptColumns

    lm = _llama()
    kept = _KeptColumns(lm._model.model, 1)
    ids = mx.array([[3, 9, 4]])
    assert kept(ids).shape == (1, 1, 32)
    assert kept.embed_tokens is lm._model.model.embed_tokens
    other = []
    t = threading.Thread(target=lambda: other.append(kept(ids).shape))
    t.start()
    t.join()
    assert other == [(1, 3, 32)]


def test_a_model_outside_the_copy_projects_every_column():
    assert not LanguageModel(_BareModel()).supports_logits_to_keep


def test_logits_to_keep_reaches_a_trunk_that_takes_the_embeddings():
    """A model whose own forward takes no embeddings runs its trunk on
    them, and the head on the kept columns."""
    import mlx.core as mx
    import numpy as np

    from gmlx._vendor.mlx_lm.models import llama

    class _IdsOnly(llama.Model):
        __module__ = "gmlx._vendor.mlx_lm.models.ids_only"

        def __call__(self, inputs, cache=None):
            return super().__call__(inputs, cache)

    base = _llama()._model
    lm = LanguageModel(_IdsOnly(base.args))
    lm._model.update(base.parameters())
    ids = mx.array([[3, 9, 4, 7, 1]])
    embeds = lm.input_embeds(mx.array([[6, 2, 8, 5, 4]]))
    full = lm(ids, inputs_embeds=embeds).logits
    kept = lm(ids, inputs_embeds=embeds, logits_to_keep=1).logits
    assert kept.shape == (1, 1, 64)
    # The full head is a TF32 GEMM on the GPU, the kept column a GEMV.
    np.testing.assert_allclose(np.array(kept), np.array(full[:, -1:]),
                               atol=1e-2)


def _synth(meta_and_shapes):
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.loader import build_model

    meta, shapes = meta_and_shapes
    return build_model(synthesize_config(meta, shapes))


def _gemma3n():
    from gmlx.load.loader import build_model

    layer_types = ["sliding_attention", "full_attention"] * 2
    return build_model({"model_type": "gemma3n", "text_config": dict(
        model_type="gemma3n_text", hidden_size=64, num_hidden_layers=4,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-6, vocab_size=32,
        vocab_size_per_layer_input=32, num_kv_shared_layers=2,
        sliding_window=512, max_position_embeddings=1024, rope_theta=1e6,
        rope_local_base_freq=1e4, final_logit_softcapping=30.0,
        layer_types=layer_types, activation_sparsity_pattern=[0.95, 0, 0, 0],
        hidden_size_per_layer_input=16, altup_num_inputs=4,
        altup_coef_clip=120.0, altup_correct_scale=True, altup_active_idx=0,
        laurel_rank=8, rope_scaling=None)})


def _text_models():
    """Every text model type the CPU tier builds: the GGUF architectures
    from synthesized configs, the owned ones as the loader leaves them, and
    gemma-3n."""
    from test_config_synth import DENSE_ARCHES, _meta_for, _meta_gemma4_size
    from test_rope_swap import ARCHES
    from tune.tiny_train_archs import CASES, build

    rows = {a: (lambda a=a: _synth(ARCHES[a]())) for a in ARCHES}
    rows.update({a: (lambda a=a: _synth((_meta_for(a), {})))
                 for a in DENSE_ARCHES if a not in rows})
    rows["gemma4"] = lambda: _synth((_meta_gemma4_size(
        "gemma4", layers=6, hidden=64, experts=None),
        {"blk.5.attn_v.weight": [16, 64]}))
    rows.update({f"owned:{n}": (lambda n=n: build(n)[:2]) for n in CASES})
    rows["gemma3n"] = _gemma3n
    return rows


_TEXT_MODELS = _text_models()


@pytest.mark.parametrize("name", sorted(_TEXT_MODELS))
def test_prefill_from_embeddings_matches_prefill_from_ids(name):
    """The batch engine prefills from get_input_embeddings and decodes from
    token ids. Both forwards of a served text model give the same logits,
    bit for bit."""
    import mlx.core as mx

    from gmlx.models.vlm_text_only import Model as TextOnlyModel
    from gmlx.serve import bridge_vlm

    model, config = _TEXT_MODELS[name]()
    mx.eval(model.parameters())
    served = TextOnlyModel(model, config=bridge_vlm._as_dict(config))
    bridge_vlm._ensure_text_embedding_probe(served, model)
    lm = served.language_model
    ids = mx.array([[1, 5, 2, 7, 3]])
    want = lm(ids, cache=lm.make_cache()).logits
    embeds = served.get_input_embeddings(ids).inputs_embeds
    got = lm(ids, cache=lm.make_cache(), inputs_embeds=embeds).logits
    assert got.shape == want.shape
    assert mx.array_equal(got, want).item()
