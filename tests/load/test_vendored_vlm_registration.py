"""A VLM model type gmlx vendors builds the gmlx class through
build_vlm_model, ahead of the package mlx-vlm ships under the same name.

mlx-vlm 0.7.3 ships qwen4_exp and glm5_next packages, and muse_glimmer since
0.6.15, each with a parameter tree the GGUF remap does not target. Models
are built from synthesized configs and never evaluated, so no weights are
allocated and no kernel runs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen4_exp")

import gmlx.load.vlm as gvlm  # noqa: E402
from gmlx.load.config_synth import synthesize_config  # noqa: E402

from test_config_synth import (  # noqa: E402
    _GLM5NEXT_SHAPES,
    _MUSE_GLIMMER_SHAPES,
    _QWEN4EXP_SHAPES,
    _glm5next_meta,
    _muse_glimmer_meta,
    _qwen4exp_meta,
)

_CLIP = {
    "clip.vision.block_count": 1,
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 64,
    "clip.vision.attention.head_count": 2,
    "clip.vision.patch_size": 14,
    "clip.vision.image_size": 448,
    "clip.vision.spatial_merge_size": 2,
}

_GMLX_MODULES = {
    "qwen4_exp": "gmlx.models.qwen4_exp.vlm_model",
    "glm5_next": "gmlx.models.glm5_next.vlm_model",
    "muse_glimmer": "gmlx.models.muse_glimmer.vlm_model",
    "deepseek_v4_vl": "gmlx.models.deepseek_v4.vlm_model",
    "deepseek_v41_vl": "gmlx.models.deepseek_v41.vlm_model",
}


def _configs() -> dict:
    """Tiny configs for the model types upstream also ships."""
    q_text = dict(synthesize_config(_qwen4exp_meta(True, True),
                                    _QWEN4EXP_SHAPES))
    qwen4_exp = {
        "model_type": "qwen4_exp",
        "text_config": q_text,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "out_hidden_size": q_text["hidden_size"], "num_heads": 2,
            "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "num_position_embeddings": 4,
            "in_channels": 3, "image_size": 64,
            "deepstack_visual_indexes": [],
        },
        "vocab_size": q_text["vocab_size"],
        "image_token_id": 20, "video_token_id": 21,
        "vision_start_token_id": 18, "vision_end_token_id": 19,
    }
    g_text = synthesize_config(_glm5next_meta(), _GLM5NEXT_SHAPES)
    glm5_next = gvlm.synthesize_vlm_config(
        "glm5_next", _glm5next_meta(), _GLM5NEXT_SHAPES,
        {**_CLIP, "clip.vision.projection_dim": g_text["hidden_size"]})
    m_text = synthesize_config(_muse_glimmer_meta(), _MUSE_GLIMMER_SHAPES)
    muse_glimmer = gvlm.synthesize_vlm_config(
        "muse_glimmer", _muse_glimmer_meta(), _MUSE_GLIMMER_SHAPES,
        {**_CLIP, "clip.vision.projection_dim": m_text["hidden_size"]})
    return {"qwen4_exp": qwen4_exp, "glm5_next": glm5_next,
            "muse_glimmer": muse_glimmer}


def _flat_caches(model):
    out = []
    for c in model.make_cache():
        out.extend(getattr(c, "caches", None) or [c])
    return out


@pytest.mark.parametrize("model_type", sorted(_GMLX_MODULES))
def test_registration_resolves_the_gmlx_module(model_type):
    from gmlx._vendor.mlx_vlm.utils import get_model_and_args

    gvlm.register_vendored_vlm_model(model_type)
    module, _ = get_model_and_args({"model_type": model_type})
    assert module.__name__ == _GMLX_MODULES[model_type]


@pytest.mark.parametrize("model_type", ["glm5_next", "muse_glimmer",
                                        "qwen4_exp"])
def test_build_vlm_model_builds_the_gmlx_class(model_type):
    model, _ = gvlm.build_vlm_model(_configs()[model_type])
    assert type(model).__module__ == _GMLX_MODULES[model_type]


def test_qwen4_exp_caches_are_the_gmlx_qsa_class():
    """apc_qsa's disk arms check for this class, so an upstream QSA cache
    in the stack would miss them."""
    import gmlx._vendor.mlx_vlm.models.cache as vlm_cache

    from gmlx.models.qwen4_exp.model import QSAKVCache

    model, _ = gvlm.build_vlm_model(_configs()["qwen4_exp"])
    qsa = [c for c in _flat_caches(model)
           if type(c).__name__ == "QSAKVCache"]
    assert qsa and all(type(c) is QSAKVCache for c in qsa)
    assert vlm_cache.QSAKVCache is QSAKVCache


_UPSTREAM_FIRST = """
import importlib, json, sys
sys.path[:0] = {paths!r}
for t in ("qwen4_exp", "glm5_next", "muse_glimmer"):
    importlib.import_module("mlx_vlm.models." + t)
premise = {{t: sys.modules["mlx_vlm.models." + t].__file__
           for t in ("qwen4_exp", "glm5_next", "muse_glimmer")}}

import mlx_vlm.models as vlm_models
import test_vendored_vlm_registration as T
from gmlx.cache.apc_pooling import install_pooling_apc_support
from gmlx.cache.apc_qsa import install_qsa_apc_support

built = {{}}
for t, cfg in T._configs().items():
    model, _ = T.gvlm.build_vlm_model(cfg)
    built[t] = [type(model).__module__,
                getattr(vlm_models, t).__name__]
    if t == "qwen4_exp":
        qsa = [c for c in T._flat_caches(model)
               if type(c).__name__ == "QSAKVCache"]

install_pooling_apc_support()
install_qsa_apc_support()
import mlx_vlm.apc as apc
arrays, metadata = {{}}, {{}}
apc.DiskBlockStore._snapshot_exact_cache_entry(
    None, qsa[0], "c0", arrays, metadata)
print(json.dumps({{"premise": premise, "built": built,
                  "qsa_cls": type(qsa[0]).__module__,
                  "kind": metadata.get("c0_kind")}}))
"""


def test_gmlx_class_wins_when_the_upstream_package_was_imported_first():
    """The graft rebinds a name upstream already bound. Runs in a fresh
    interpreter, since importing the upstream packages patches mlx-vlm's
    processor registry for the rest of the process."""
    here = Path(__file__).resolve().parent
    paths = [str(here), str(here.parent / "load")]
    script = _UPSTREAM_FIRST.format(paths=paths)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=dict(os.environ), timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])

    for t, path in out["premise"].items():
        assert path.endswith(f"/mlx_vlm/models/{t}/__init__.py"), (t, path)
    for t, (model_module, bound) in out["built"].items():
        assert model_module == _GMLX_MODULES[t]
        assert bound == _GMLX_MODULES[t]
    assert out["qsa_cls"] == "gmlx.models.qwen4_exp.model"
    assert out["kind"] == "kq_qsa"
