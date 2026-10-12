"""The vendored text-only adapter (gmlx.models.vlm_text_only, from mlx-vlm 0.6.4
models/text_only.py). Vendoring moved the module out of mlx_vlm.models, so
package-relative imports the original relied on must have been rewritten;
the make_cache fallback's `.cache` was missed, and every served request
against an mlx-lm model without its own make_cache died in
make_prompt_cache (ModuleNotFoundError: gmlx.cache)."""
import mlx.nn as nn

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
