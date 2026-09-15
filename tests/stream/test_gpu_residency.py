#!/usr/bin/env python3
"""The Metal residency-set install (gmlx.stream.expert_streaming
._install_gpu_residency): it runs after the pin, the arena and the ring,
so it is the step that can push their sum past the working set."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.stream.expert_streaming as es

MB = 1 << 20


class _Fake:
    """mlx_kquant's residency ops, recording what was inserted."""

    def __init__(self):
        self.inserted = []

    def residency_insert(self, a):
        self.inserted.append(a)
        return True

    def residency_commit(self):
        pass


@pytest.fixture
def kq(monkeypatch):
    fake = _Fake()
    import mlx_kquant
    monkeypatch.setattr(mlx_kquant, "residency_insert", fake.residency_insert,
                        raising=False)
    monkeypatch.setattr(mlx_kquant, "residency_commit", fake.residency_commit,
                        raising=False)
    return fake


def _model(n=4, dims=512):
    """n float32 leaves of 1 MB each."""
    m = nn.Sequential(*[nn.Linear(dims, dims, bias=False) for _ in range(n)])
    mx.eval(m.parameters())
    return m


def test_no_room_argument_inserts_everything(kq):
    es._install_gpu_residency(_model(), {})
    assert len(kq.inserted) == 4


def test_room_caps_the_set(kq, capsys):
    es._install_gpu_residency(_model(), {}, room=2 * MB + 1)
    assert len(kq.inserted) == 2
    out = capsys.readouterr().out
    assert "2 buffers" in out and "left out" in out


def test_zero_room_inserts_nothing(kq):
    es._install_gpu_residency(_model(), {}, room=0)
    assert kq.inserted == []
