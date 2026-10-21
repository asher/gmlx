"""Serve prefills a model that streams its experts in the wider streamed
chunk, as the CLI does, unless a prefill step was set."""

from types import SimpleNamespace

import pytest

import gmlx._vendor.mlx_vlm.server.generation as gen
from gmlx.load.loader import _STREAMING_PREFILL_STEP


class _NoThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def _generator(monkeypatch, **kwargs):
    monkeypatch.setattr(gen, "Thread", _NoThread)
    return gen.ResponseGenerator(model_path="unused", **kwargs)


def _vlm(streamed: bool, model_type: str = "muse_spark"):
    # moe_streaming_active reads _kq_cpu_only off the text tower's modules.
    layer = SimpleNamespace(
        modules=lambda: [SimpleNamespace(_kq_cpu_only=streamed)])
    lm = SimpleNamespace(model_type=model_type, layers=[layer])
    return SimpleNamespace(language_model=lm)


@pytest.fixture(autouse=True)
def _no_env_step(monkeypatch):
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)


def test_model_load_adopts_the_streamed_step(monkeypatch):
    rg = _generator(monkeypatch)
    assert rg.prefill_step_size == gen.DEFAULT_PREFILL_STEP_SIZE
    vlm = _vlm(streamed=True)
    monkeypatch.setattr(gen, "load_model_resources", lambda path, adapter: (
        vlm, SimpleNamespace(tokenizer=SimpleNamespace()),
        SimpleNamespace(eos_token_id=None)))
    monkeypatch.setattr(gen, "resolve_eos_token_ids", lambda *a, **k: [])
    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    rg._initialize_model()
    assert rg.prefill_step_size == _STREAMING_PREFILL_STEP
    assert rg._effective_prefill_step_size() == _STREAMING_PREFILL_STEP
    # an arch profile's step replaces it, as it replaces the stock one
    assert vlm.language_model._gmlx_adopted_prefill_step == \
        _STREAMING_PREFILL_STEP


def test_resident_model_keeps_the_default_step(monkeypatch):
    rg = _generator(monkeypatch)
    rg._adopt_streamed_prefill_step(_vlm(streamed=False))
    assert rg.prefill_step_size == gen.DEFAULT_PREFILL_STEP_SIZE


def test_a_resident_cpu_expert_codec_keeps_the_default_step(monkeypatch):
    # Its experts run on the CPU from RAM, marked like streamed ones.
    rg = _generator(monkeypatch)
    vlm = _vlm(streamed=True)
    vlm.language_model._kq_streaming = False
    rg._adopt_streamed_prefill_step(vlm)
    assert rg.prefill_step_size == gen.DEFAULT_PREFILL_STEP_SIZE


def _text_gguf(streamed: bool):
    """A text GGUF as serve loads it: the install marks the raw model, and
    the text-only wrapper and its adapter carry no verdict. The expert
    modules are marked CPU-only either way, as a resident CPU codec is."""
    import mlx.nn as nn

    from gmlx.models.vlm_text_only import Model

    class Experts(nn.Module):
        pass

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Experts()

    class Raw(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]
            self.model_type = "qwen3_moe"

    raw = Raw()
    raw.layers[0].mlp._kq_cpu_only = True
    raw._kq_streaming = streamed
    return Model(raw, config={"model_type": "qwen3_moe"})


@pytest.mark.parametrize("streamed", [True, False])
def test_a_text_gguf_reads_the_install_verdict(monkeypatch, streamed):
    from gmlx.stream.expert_streaming import experts_streamed

    model = _text_gguf(streamed)
    rg = _generator(monkeypatch)
    rg._adopt_streamed_prefill_step(model)
    assert rg.prefill_step_size == (
        _STREAMING_PREFILL_STEP if streamed else gen.DEFAULT_PREFILL_STEP_SIZE)
    # The batch holds the language model.
    assert experts_streamed(model.language_model) is streamed


def test_a_step_from_the_environment_wins(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    rg = _generator(monkeypatch)
    rg._adopt_streamed_prefill_step(_vlm(streamed=True))
    assert rg.prefill_step_size == 2048


def test_a_step_argument_wins(monkeypatch):
    rg = _generator(monkeypatch, prefill_step_size=1024)
    rg._adopt_streamed_prefill_step(_vlm(streamed=True))
    assert rg.prefill_step_size == 1024


def test_the_model_type_entry_applies(monkeypatch):
    from gmlx.load.loader import _STREAMING_PREFILL_STEP_BY_MODEL_TYPE

    rg = _generator(monkeypatch)
    rg._adopt_streamed_prefill_step(_vlm(streamed=True, model_type="hy_v4"))
    assert rg.prefill_step_size == _STREAMING_PREFILL_STEP_BY_MODEL_TYPE[
        "hy_v4"]
