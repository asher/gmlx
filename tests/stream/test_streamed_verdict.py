"""experts_streamed walks a model without an install verdict once, and the
install's verdict wins over the walk."""

from __future__ import annotations

from types import SimpleNamespace

from gmlx.stream import expert_streaming as es


class _Layer:
    def __init__(self, walks, cpu_only=False):
        self._walks = walks
        self._cpu_only = cpu_only

    def modules(self):
        self._walks.append(self)
        return [SimpleNamespace(_kq_cpu_only=self._cpu_only)]


def _model(walks, cpu_only=False, layers=3):
    return SimpleNamespace(layers=[_Layer(walks, cpu_only) for _ in range(layers)])


def test_a_model_without_a_verdict_is_walked_once():
    walks = []
    model = _model(walks)
    assert es.experts_streamed(model) is False
    assert len(walks) == 3
    assert es.experts_streamed(model) is False
    assert es.experts_streamed(model) is False
    assert len(walks) == 3


def test_the_walk_reads_a_cpu_only_codec_as_streamed():
    walks = []
    assert es.experts_streamed(_model(walks, cpu_only=True)) is True


def test_the_installs_verdict_wins_over_the_kept_walk():
    walks = []
    model = _model(walks)
    assert es.experts_streamed(model) is False
    model._kq_streaming = True
    assert es.experts_streamed(model) is True
    assert len(walks) == 3


def test_the_walk_is_kept_on_the_outer_model_of_a_wrapper_chain():
    walks = []
    inner = _model(walks)
    outer = SimpleNamespace(language_model=inner)
    assert es.experts_streamed(outer) is False
    assert es.experts_streamed(outer) is False
    assert len(walks) == 3
    assert not hasattr(outer, "_kq_streaming")
    assert not hasattr(inner, "_kq_streaming")


def test_no_model_is_not_streamed():
    assert es.experts_streamed(None) is False
