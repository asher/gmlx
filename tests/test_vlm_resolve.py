"""resolve_vlm_model_type: the (LLM arch, mmproj projector) dispatch table.

CPU-only - feeds synthetic mmproj metadata dicts; no GGUFs, no mlx-vlm import.
Pins two properties: every advertised family resolves to the model_type the
downstream remap/config/processor surfaces implement, and unsupported pairings
fail *here*, by name, not three stages later at the remap gate.
"""
import pytest

from gmlx.vlm import UnsupportedVLMError, resolve_vlm_model_type


@pytest.mark.parametrize("llm_arch,mm_meta,expected", [
    ("llama", {"clip.has_llava_projector": True}, "llava"),
    ("llama", {"clip.projector_type": "pixtral"}, "pixtral"),
    ("qwen35", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_5"),
    ("qwen35moe", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_5_moe"),
    ("qwen3vlmoe", {"clip.vision.projector_type": "qwen3vl_merger"}, "qwen3_omni_moe"),
    ("gemma4", {"clip.vision.projector_type": "gemma4v"}, "gemma4"),
    ("gemma4", {"clip.vision.projector_type": "gemma4uv"}, "gemma4_unified"),
])
def test_supported_families_resolve(llm_arch, mm_meta, expected):
    assert resolve_vlm_model_type(llm_arch, mm_meta) == expected


def test_qwen2vl_fails_early_with_family_named():
    # qwen2_vl has no remap/config/processor implementation; the resolver must
    # refuse it up front instead of handing back a model_type that dies later.
    with pytest.raises(UnsupportedVLMError, match="Qwen2-VL"):
        resolve_vlm_model_type("qwen2vl", {"clip.projector_type": "qwen2vl_merger"})


def test_unknown_pairing_raises_with_both_names():
    with pytest.raises(UnsupportedVLMError, match="mystery.*whatproj|whatproj"):
        resolve_vlm_model_type("mystery", {"clip.projector_type": "whatproj"})
