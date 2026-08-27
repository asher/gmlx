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
    from mlx_vlm.models.cache import KVCache

    caches = LanguageModel(_BareModel()).make_cache()
    assert len(caches) == 2
    assert all(isinstance(c, KVCache) for c in caches)


def test_make_cache_prefers_the_models_own():
    class _Own(nn.Module):
        layers = [object()]

        def make_cache(self):
            return ["own"]

    assert LanguageModel(_Own()).make_cache() == ["own"]
