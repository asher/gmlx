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


def test_streamed_model_takes_the_streamed_step(monkeypatch):
    rg = _generator(monkeypatch)
    assert rg.prefill_step_size == gen.DEFAULT_PREFILL_STEP_SIZE
    rg._adopt_streamed_prefill_step(_vlm(streamed=True))
    assert rg.prefill_step_size == _STREAMING_PREFILL_STEP
    assert rg._effective_prefill_step_size() == _STREAMING_PREFILL_STEP


def test_resident_model_keeps_the_default_step(monkeypatch):
    rg = _generator(monkeypatch)
    rg._adopt_streamed_prefill_step(_vlm(streamed=False))
    assert rg.prefill_step_size == gen.DEFAULT_PREFILL_STEP_SIZE


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
