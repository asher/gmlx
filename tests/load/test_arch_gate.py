#!/usr/bin/env python3
"""The load gate: arch -> mlx-lm model_type resolution and its three refusals.

Pure table logic - no tensors, no GPU. Mirrors the gate's contract: a GGUF is
loadable iff (a) its arch maps to an mlx-lm model_type, (b) the installed
mlx-lm ships that model, and (c) a config synthesizer exists OR an hf_source is
supplied. Each refusal must name what's wrong and what to do.
"""

from __future__ import annotations

import pytest

from gmlx import arch_table, config_synth  # noqa: E402
from gmlx.arch_table import UnsupportedArchError, gate  # noqa: E402


def test_supported_arch_resolves():
    e = gate("qwen2")
    assert e.gguf_arch == "qwen2"
    assert e.model_type == "qwen2"
    assert e.remap_alias == "QWEN2"


def test_every_synth_arch_gates_without_hf_source():
    # The single source of truth for "loadable with no override".
    for arch in config_synth.supported_arches():
        e = gate(arch)  # must not raise
        assert e.model_type == config_synth.GGUF_ARCH_TO_MODEL_TYPE[arch]


def test_gemma_embedding_gates_via_mlx_embeddings_backend():
    # EmbeddingGemma is built from mlx-embeddings (not mlx-lm), reusing the
    # gemma3 backbone remap; the gate must accept it on the external-class path.
    e = gate("gemma-embedding")
    assert e.model_type == "gemma_embedding"
    assert e.remap_alias == "GEMMA3"
    assert e.backend == "mlx-embeddings"


def test_unknown_arch_refused_mode_a():
    with pytest.raises(UnsupportedArchError) as ei:
        gate("not-a-real-arch")
    msg = str(ei.value)
    assert "no mapping" in msg
    # lists the mapped archs so the user can see what IS covered.
    assert "qwen2" in msg


def test_mapped_but_no_synth_refused_mode_c(monkeypatch):
    # An arch that's mapped to an mlx-lm model_type but whose synthesizer hasn't
    # landed yet must demand hf_source (rather than emit a half-built config).
    # Every shipped arch currently has a synthesizer, so simulate the gap by
    # making the synth probe report "missing" for this gate call.
    monkeypatch.setattr(arch_table, "has_synth", lambda arch: False)
    with pytest.raises(UnsupportedArchError) as ei:
        gate("qwen3moe")
    assert "config synthesizer" in str(ei.value)
    assert "hf_source" in str(ei.value)


def test_hf_source_bypasses_missing_synth():
    # Same arch, with an override supplied -> gate passes (config comes from
    # the user's config.json, not the synthesizer).
    e = gate("qwen3moe", hf_source="some/hf-repo")
    assert e.model_type == "qwen3_moe"


def test_missing_mlx_lm_model_refused_mode_b(monkeypatch):
    monkeypatch.setattr(arch_table, "mlx_lm_has_model", lambda mt: False)
    with pytest.raises(UnsupportedArchError) as ei:
        gate("qwen2")
    assert "no" in str(ei.value) and "mlx_lm/models/qwen2.py" in str(ei.value)


def test_load_config_from_source_local_dir(tmp_path):
    # A local dir with a config.json is used directly - no network.
    pytest.importorskip("mlx_lm")
    from gmlx.loader import _load_config_from_source

    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen2", "hidden_size": 64}')
    cfg = _load_config_from_source(str(tmp_path))
    assert cfg == {"model_type": "qwen2", "hidden_size": 64}


def test_load_config_from_source_hf_id(tmp_path, monkeypatch):
    # A non-dir source is treated as an HF repo id: config.json is fetched
    # via hf_hub_download (stubbed here - the test stays offline).
    pytest.importorskip("mlx_lm")
    import huggingface_hub

    from gmlx.loader import _load_config_from_source

    fetched = tmp_path / "config.json"
    fetched.write_text('{"model_type": "llama"}')
    calls = {}

    def fake_download(repo_id, filename):
        calls["args"] = (repo_id, filename)
        return str(fetched)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cfg = _load_config_from_source("some-org/some-repo")
    assert cfg == {"model_type": "llama"}
    assert calls["args"] == ("some-org/some-repo", "config.json")


def test_disabled_arch_refused_even_with_hf_source():
    # An arch whose synth/remap/loader are complete but that has no
    # known-good GGUF is gate-disabled. The refusal is intentional and NOT
    # bypassable by an hf_source override (the defect is in the weights, not the
    # config), and the arch is excluded from the no-override loadable set.
    assert config_synth.DISABLED_ARCHES, "expected at least one disabled arch"
    for arch in config_synth.DISABLED_ARCHES:
        assert arch not in config_synth.supported_arches()
        for kwargs in ({}, {"hf_source": "some/hf-repo"}):
            with pytest.raises(UnsupportedArchError) as ei:
                gate(arch, **kwargs)
            assert "disabled" in str(ei.value).lower()
