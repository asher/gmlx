"""The model types gmlx owns resolve to the gmlx modules through the tables
in gmlx.models.owned, which the vendored loaders read before their own
packages.

mlx-vlm 0.7.3 ships qwen4_exp, glm5_next and muse_glimmer packages, each
with a parameter tree the GGUF remap does not target, and the copy leaves
them out. Models are built from synthesized configs and never evaluated, so
no weights are allocated and no kernel runs.
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gmlx.load.vlm as gvlm
from gmlx.load.config_synth import synthesize_config
from gmlx.models.owned import OWNED_LM_MODELS, OWNED_VLM_MODELS

from test_config_synth import (  # noqa: E402
    _GLM5NEXT_SHAPES,
    _MUSE_GLIMMER_SHAPES,
    _MUSE_SPARK_SHAPES,
    _QWEN4EXP_SHAPES,
    _glm5next_meta,
    _muse_glimmer_meta,
    _muse_spark_meta,
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


def _configs() -> dict:
    """Tiny configs for the owned VLM types mlx-vlm also ships."""
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
    s_text = synthesize_config(_muse_spark_meta(), _MUSE_SPARK_SHAPES)
    muse_spark = gvlm.synthesize_vlm_config(
        "muse_spark", _muse_spark_meta(), _MUSE_SPARK_SHAPES,
        {**_CLIP, "clip.vision.projection_dim": s_text["hidden_size"]})
    return {"qwen4_exp": qwen4_exp, "glm5_next": glm5_next,
            "muse_glimmer": muse_glimmer, "muse_spark": muse_spark}


def _flat_caches(model):
    out = []
    for c in model.make_cache():
        out.extend(getattr(c, "caches", None) or [c])
    return out


@pytest.mark.parametrize("model_type", sorted(OWNED_LM_MODELS))
def test_lm_table_resolves_the_gmlx_module(model_type):
    from gmlx._vendor.mlx_lm.utils import _get_classes

    owned = importlib.import_module(OWNED_LM_MODELS[model_type])
    model, args = _get_classes({"model_type": model_type})
    assert model is owned.Model and args is owned.ModelArgs


@pytest.mark.parametrize("model_type", sorted(OWNED_VLM_MODELS))
def test_vlm_table_resolves_the_gmlx_module(model_type):
    from gmlx._vendor.mlx_vlm.utils import get_model_and_args

    module, resolved = get_model_and_args({"model_type": model_type})
    assert module.__name__ == OWNED_VLM_MODELS[model_type]
    assert resolved == model_type


def test_the_copy_ships_no_package_under_an_owned_type():
    """An owned type never enters a vendored namespace, so the copy holds no
    module of that name for a later import to find beside the gmlx one."""
    shipped = [f"gmlx._vendor.mlx_lm.models.{t}" for t in OWNED_LM_MODELS]
    shipped += [f"gmlx._vendor.mlx_vlm.models.{t}" for t in OWNED_VLM_MODELS]
    shipped += [f"gmlx._vendor.mlx_vlm.speculative.drafters.{t}_mtp"
                for t in OWNED_VLM_MODELS]
    assert [m for m in shipped if importlib.util.find_spec(m)] == []


def test_every_mtp_splitter_row_names_a_module_of_the_copy():
    from gmlx._vendor.mlx_vlm.speculative.drafters import (
        DRAFTER_KIND_BY_MODEL_TYPE,
    )
    from gmlx._vendor.mlx_vlm.speculative.drafters.mtp_split import (
        MTP_SPLITTERS,
    )

    missing = [ref for ref in MTP_SPLITTERS.values()
               if importlib.util.find_spec(ref.partition(":")[0]) is None]
    assert missing == []
    assert not {f"{t}_mtp" for t in OWNED_VLM_MODELS} & set(
        DRAFTER_KIND_BY_MODEL_TYPE)


@pytest.mark.parametrize("model_type", ["glm5_next", "muse_glimmer",
                                        "muse_spark", "qwen4_exp"])
def test_build_vlm_model_builds_the_gmlx_class(model_type):
    model, _ = gvlm.build_vlm_model(_configs()[model_type])
    assert type(model).__module__ == OWNED_VLM_MODELS[model_type]


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


_QSA_DISK_KIND = """
import json, sys
sys.path[:0] = {paths!r}
import test_owned_model_routing as T
from gmlx.cache.apc_pooling import install_pooling_apc_support
from gmlx.cache.apc_qsa import install_qsa_apc_support
import gmlx._vendor.mlx_vlm.apc as apc

model, _ = T.gvlm.build_vlm_model(T._configs()["qwen4_exp"])
qsa = [c for c in T._flat_caches(model) if type(c).__name__ == "QSAKVCache"]
install_pooling_apc_support()
install_qsa_apc_support()
arrays, metadata = {{}}, {{}}
apc.DiskBlockStore._snapshot_exact_cache_entry(
    None, qsa[0], "c0", arrays, metadata)
print(json.dumps({{"kind": metadata.get("c0_kind")}}))
"""


def test_the_qwen4_exp_qsa_cache_takes_the_gmlx_disk_arm():
    """The QSA disk arm snapshots the cache the owned VLM builds. Runs in a
    fresh interpreter, since the installers patch the APC store for the
    rest of the process."""
    here = Path(__file__).resolve().parent
    script = _QSA_DISK_KIND.format(paths=[str(here)])
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=dict(os.environ), timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["kind"] == "kq_qsa"
