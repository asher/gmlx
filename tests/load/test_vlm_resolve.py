"""resolve_vlm_model_type: the (LLM arch, mmproj projector) dispatch table.

CPU-only - feeds synthetic mmproj metadata dicts; no GGUFs, no mlx-vlm import.
Pins two properties: every advertised family resolves to the model_type the
downstream remap/config/processor surfaces implement, and unsupported pairings
fail *here*, by name, not three stages later at the remap gate.
"""
import pytest

from gmlx.load.vlm import UnsupportedVLMError, resolve_vlm_model_type


@pytest.mark.parametrize("llm_arch,mm_meta,expected", [
    ("llama", {"clip.has_llava_projector": True}, "llava"),
    ("llama", {"clip.projector_type": "pixtral"}, "pixtral"),
    ("qwen35", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_5"),
    ("qwen35moe", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_5_moe"),
    ("qwen3vlmoe", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_omni_moe"),
    ("gemma4", {"clip.vision.projector_type": "gemma4v"}, "gemma4"),
    ("gemma4", {"clip.vision.projector_type": "gemma4uv"}, "gemma4_unified"),
    ("muse-glimmer", {"clip.projector_type": "muse-glimmer"}, "muse_glimmer"),
    ("muse-glimmer", {"clip.vision.projector_type": "muse-glimmer"},
     "muse_glimmer"),
    ("deepseek2", {"clip.projector_type": "kimik25"}, "kimi_k25"),
    ("deepseek4", {"clip.projector_type": "deepseek4v"}, "deepseek_v4_vl"),
    ("qwen3vl", {"clip.projector_type": "qwen3vl_merger"}, "qwen3_vl"),
    ("qwen3vl", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_vl"),
    ("qwen35", {"clip.projector_type": "qwen3vl_merger"}, "qwen3_5"),
    ("qwen35moe", {"clip.projector_type": "qwen3vl_merger"}, "qwen3_5_moe"),
    ("qwen4exp", {"clip.projector_type": "qwen3vl_merger"}, "qwen4_exp"),
])
def test_supported_families_resolve(llm_arch, mm_meta, expected):
    assert resolve_vlm_model_type(llm_arch, mm_meta) == expected


def test_one_projector_resolves_by_the_text_arch():
    """Qwen3-VL and Qwen3.6 mmprojs name the same projector. The text arch
    picks the container: only qwen3vl gets the one that reads the deepstack
    features, and a Qwen3.6 pair resolves as it did before that row."""
    mm = {"clip.projector_type": "qwen3vl_merger"}
    assert resolve_vlm_model_type("qwen3vl", mm) == "qwen3_vl"
    assert resolve_vlm_model_type("qwen35", mm) == "qwen3_5"
    assert resolve_vlm_model_type("qwen35moe", mm) == "qwen3_5_moe"


def test_kimik25_on_a_foreign_text_arch_is_refused():
    """GLM-5.2-V ships the same vision encoder + projector as Kimi-K2.5 under a
    different text arch. Resolving it to kimi_k25 would build a Kimi text tower
    for GLM weights, so the pairing has to fail by name."""
    with pytest.raises(UnsupportedVLMError, match="kimik25.*glm-dsa|GLM-5.2-V"):
        resolve_vlm_model_type("glm-dsa", {"clip.projector_type": "kimik25"})


def test_qwen2vl_fails_early_with_family_named():
    # qwen2_vl has no remap/config/processor implementation; the resolver must
    # refuse it up front instead of handing back a model_type that dies later.
    with pytest.raises(UnsupportedVLMError, match="Qwen2-VL"):
        resolve_vlm_model_type("qwen2vl", {"clip.projector_type": "qwen2vl_merger"})


def test_unknown_pairing_raises_with_both_names():
    with pytest.raises(UnsupportedVLMError, match="mystery.*whatproj|whatproj"):
        resolve_vlm_model_type("mystery", {"clip.projector_type": "whatproj"})


def test_deepseek4v_on_a_foreign_text_arch_is_refused():
    """The deepseek4v aligner feeds a text tower that expands image blocks
    and routes them with its own bias; only the deepseek4 arch has that."""
    with pytest.raises(UnsupportedVLMError, match="deepseek4v.*deepseek2"):
        resolve_vlm_model_type("deepseek2", {"clip.projector_type": "deepseek4v"})


def test_the_qwen3vl_container_takes_only_the_qwen3vl_text_arch():
    """A plain qwen3 text file beside this projector does not get the
    container that reads the deepstack features: its header has no rotary
    axis widths and no deepstack count. It resolves as it did before the
    qwen3vl row."""
    mm = {"clip.projector_type": "qwen3vl_merger"}
    assert resolve_vlm_model_type("qwen3", mm) == "qwen3_5"
    assert resolve_vlm_model_type("qwen3moe", mm) == "qwen3_5"
