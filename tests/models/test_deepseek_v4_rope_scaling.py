"""DeepSeek V4 and V4.1 rope scaling: compressed layers rotate with the
resolved table on the compress base and no magnitude (llama.cpp's
dsv4_rope_attn_factor cancels it), uncompressed layers with the run's
base and no scaling, and a yarn block missing keys builds with
llama.cpp's defaults."""

import numpy as np
import pytest

from gmlx.load import rope_scaling as rs
from gmlx.load.config_synth import synthesize_config
from gmlx.load.rope_scaling import RopeOverride
from gmlx.models.deepseek_v4 import model as v4
from test_config_synth import _DEEPSEEK4_SHAPES, _deepseek4_meta


def _inv(rope):
    return 1.0 / np.asarray(rope._freqs, dtype=np.float64)


def test_resolved_table_matches_the_class_table_for_the_header():
    """The header path through the resolver keeps the class's yarn table."""
    meta = _deepseek4_meta()
    s = rs.resolve(meta, "deepseek4")
    hf = {"type": "yarn", "factor": 16.0,
          "original_max_position_embeddings": 512,
          "beta_fast": 32.0, "beta_slow": 1.0}
    a = v4.DeepseekV4RoPE(8, 160000.0, hf, 1024)
    b = v4.DeepseekV4RoPE(8, 160000.0, s, 1024)
    np.testing.assert_allclose(_inv(b), _inv(a), rtol=2e-6)
    np.testing.assert_allclose(_inv(b), rs.inv_freq(8, 160000.0, s),
                               rtol=1e-6)


def test_a_yarn_block_missing_keys_builds():
    """factor and original_max_position_embeddings were indexed without a
    default, so a yarn block missing either raised."""
    r = v4.DeepseekV4RoPE(8, 10000.0, {"type": "yarn"}, 4096)
    plain = v4.DeepseekV4RoPE(8, 10000.0, None, 4096)
    np.testing.assert_allclose(_inv(r), _inv(plain), rtol=1e-6)
    v4.DeepseekV4RoPE(8, 10000.0, {"type": "yarn", "factor": 4.0}, 4096)


def _model(override):
    v4.ensure_registered()
    cfg = synthesize_config(_deepseek4_meta(), tensor_shapes=_DEEPSEEK4_SHAPES,
                            rope_override=override)
    args = v4.ModelArgs.from_dict(cfg)
    return v4.Model(args), cfg


def _layer_ropes(model):
    out = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.attn if hasattr(layer, "attn") else layer.self_attn
        out.append((i, attn))
    return out


@pytest.mark.parametrize("override", [
    None,
    RopeOverride(rope_scaling="yarn", rope_scale=4.0, yarn_orig_ctx=1024),
    RopeOverride(rope_scaling="yarn", rope_scale=4.0, rope_freq_base=50000.0),
], ids=["header", "factor-4", "freq-base"])
def test_layers_follow_llama_cpp(override):
    model, cfg = _model(override)
    s = rs.from_config(cfg)
    ratios = cfg["compress_ratios"]
    for i, attn in _layer_ropes(model):
        if ratios[i]:
            np.testing.assert_allclose(
                _inv(attn.rope), rs.inv_freq(8, 160000.0, s), rtol=1e-6)
            comp = getattr(attn, "compressor", None)
            if comp is not None and hasattr(comp, "rope"):
                np.testing.assert_allclose(
                    _inv(comp.rope), rs.inv_freq(8, 160000.0, s), rtol=1e-6)
        else:
            base = (override.rope_freq_base if override is not None
                    and override.rope_freq_base else 10000.0)
            np.testing.assert_allclose(
                _inv(attn.rope), rs.inv_freq(8, base), rtol=1e-6)
    if override is not None:
        assert s.factor == 4.0


@pytest.mark.parametrize("arch", ["deepseek4", "deepseek41"])
def test_no_options_keep_the_class_table(arch):
    """Without run options every rope of the model is bit for bit the one
    the class builds from the config's own yarn block."""
    from gmlx.models.deepseek_v41 import model as v41
    from test_config_synth import (_DEEPSEEK41_SHAPES, _deepseek41_meta)

    mod, meta, shapes = ((v4, _deepseek4_meta(), _DEEPSEEK4_SHAPES)
                         if arch == "deepseek4" else
                         (v41, _deepseek41_meta(), _DEEPSEEK41_SHAPES))
    mod.ensure_registered()
    cfg = synthesize_config(meta, tensor_shapes=shapes)
    assert rs.from_config(cfg).active
    plain = {k: v for k, v in cfg.items() if k != rs.CONFIG_KEY}
    # The class block itself reaches the rope: on the CPU the two tables
    # can round alike, on the GPU they do not.
    assert v4.rope_scaling_for(mod.ModelArgs.from_dict(cfg)) == cfg["rope_scaling"]

    def ropes(c):
        m = mod.Model(mod.ModelArgs.from_dict(c))
        return [r for _, r in m.named_modules()
                if isinstance(r, v4.DeepseekV4RoPE)]

    a, b = ropes(cfg), ropes(plain)
    assert a and len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x._freqs), np.asarray(y._freqs))


def test_deepseek_is_wired():
    assert {"deepseek4", "deepseek41"} <= rs.WIRED_ARCHES
