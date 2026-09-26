"""A server request with no output cap runs until EOS or the context fills."""
from __future__ import annotations

import types

import pytest

pytest.importorskip("mlx_vlm.server.generation")

import gmlx.serve.patches.sampling as sampling  # noqa: E402
import gmlx.serve.patches.api_contract as api_contract  # noqa: E402


def test_cap_fills_the_context():
    assert sampling.until_eos_cap(1000, 8192, 2048) == 7192
    assert sampling.until_eos_cap(9000, 8192, 2048) == 1
    assert sampling.until_eos_cap(1000, None, 2048) == 2048


def test_native_context_reads_the_text_config():
    model = types.SimpleNamespace(config=types.SimpleNamespace(
        text_config=types.SimpleNamespace(max_position_embeddings=32768)))
    assert sampling.native_context(model) == 32768
    assert sampling.native_context(types.SimpleNamespace()) is None


def test_budget_check_counts_only_the_prompt_for_the_default(monkeypatch):
    gen = pytest.importorskip("mlx_vlm.server.generation")
    monkeypatch.setattr(gen, "get_configured_context_limit", lambda: 4096)
    api_contract._check_context_budget(4000, sampling.UNTIL_EOS)
    with pytest.raises(gen.PromptTooLongError):
        api_contract._check_context_budget(4000, 200)
    with pytest.raises(gen.PromptTooLongError):
        api_contract._check_context_budget(5000, sampling.UNTIL_EOS)


def test_hook_swaps_the_marker_for_the_room_left(monkeypatch):
    gen = pytest.importorskip("mlx_vlm.server.generation")

    class FakeGen:
        model = types.SimpleNamespace(config=types.SimpleNamespace(
            max_position_embeddings=10000))

        def _make_thinking_budget_criteria(self, args, input_ids):
            return "orig"

    monkeypatch.setattr(gen, "ResponseGenerator", FakeGen)
    monkeypatch.setattr(sampling.serving, "get_active_spec", lambda: None)
    monkeypatch.setattr(gen, "get_configured_context_limit", lambda: None)
    sampling.install_until_eos_default()
    args = types.SimpleNamespace(max_tokens=sampling.UNTIL_EOS)
    assert FakeGen()._make_thinking_budget_criteria(args, list(range(400))) \
        == "orig"
    assert args.max_tokens == 9600
    pinned = types.SimpleNamespace(max_tokens=512)
    FakeGen()._make_thinking_budget_criteria(pinned, [1, 2, 3])
    assert pinned.max_tokens == 512


def test_serve_sets_the_marker_unless_capped(monkeypatch):
    import os
    import gmlx.serve.server as srv
    from tests.serve.test_server import _ns, _one_model_cfg, _stub_serving_stack

    _stub_serving_stack(monkeypatch)
    assert srv._serve(_one_model_cfg(), _ns(), None) == 0
    assert os.environ["MLX_VLM_MAX_TOKENS"] == str(sampling.UNTIL_EOS)
    _stub_serving_stack(monkeypatch)
    assert srv._serve(_one_model_cfg(), _ns(max_tokens=300), None) == 0
    assert os.environ["MLX_VLM_MAX_TOKENS"] == "300"
