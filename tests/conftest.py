"""Shared pytest fixtures.

The logic tests (remap / config-synth / arch-gate / transforms / preflight) are
pure Python and need no models - they run anywhere, including CI.

The ``integration`` tests need real GGUFs, and the parity tests additionally
need a ``llama.cpp`` completion binary. To keep this package free of any machine-
specific paths, those locations come from the environment and the tests skip when
unset:

  KQUANT_TEST_GGUF_DIR   directory searched recursively for ``*.gguf`` test models
  KQUANT_LLAMACPP_BIN    path to a llama.cpp ``llama-completion`` (or ``llama-cli``)
                         binary, used as the reference for greedy/long-context parity
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    # The logic tests touch only small mx array ops (transforms,
    # config-synth instantiation) and never dispatch a kquant kernel, so they
    # can run on the CPU device. Set KQUANT_FORCE_CPU=1 in environments without
    # a usable Metal GPU (e.g. CI) to keep them off the GPU path.
    if os.environ.get("KQUANT_FORCE_CPU"):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)


@pytest.fixture(scope="session")
def gguf_dir() -> Path:
    d = os.environ.get("KQUANT_TEST_GGUF_DIR")
    if not d:
        pytest.skip("set KQUANT_TEST_GGUF_DIR to a dir of *.gguf to run integration tests")
    p = Path(d).expanduser()
    if not p.is_dir():
        pytest.skip(f"KQUANT_TEST_GGUF_DIR={d!r} is not a directory")
    return p


@pytest.fixture(scope="session")
def llamacpp_bin() -> str:
    b = os.environ.get("KQUANT_LLAMACPP_BIN")
    if not b:
        pytest.skip("set KQUANT_LLAMACPP_BIN to a llama.cpp completion binary for parity tests")
    if not Path(b).expanduser().is_file():
        pytest.skip(f"KQUANT_LLAMACPP_BIN={b!r} not found")
    b = str(Path(b).expanduser())
    # Parity tests drive one-shot completions via -no-cnv. Newer llama.cpp
    # builds split that out of llama-cli (interactive-only) into
    # llama-completion; catch the wrong binary here, not 20 minutes into a run.
    import subprocess
    try:
        help_text = subprocess.run(
            [b, "--help"], capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:
        pytest.skip(f"KQUANT_LLAMACPP_BIN={b!r} --help failed: {e}")
    if "-no-cnv" not in help_text:
        pytest.fail(
            f"KQUANT_LLAMACPP_BIN={b!r} does not support -no-cnv (interactive-"
            f"only llama-cli?) - point it at llama-completion instead")
    return b


@pytest.fixture(scope="session")
def gguf_index(gguf_dir) -> dict:
    """Map ``general.architecture`` -> list of GGUF paths found under the test dir.

    Reads only the header of each shard-0 GGUF (cheap). Split shards collapse to
    their first shard. Lets a parity test ask for "a gemma2 model" without hard-
    coding filenames.
    """
    from gmlx.headerscan import scan_gguf

    index: dict[str, list[str]] = {}
    for path in sorted(gguf_dir.rglob("*.gguf")):
        name = path.name
        # skip non-first split shards and multimodal projectors
        if "-of-" in name and not ("00001-of-" in name or "-00001-of" in name):
            continue
        if "mmproj" in name:
            continue
        try:
            kv = scan_gguf(str(path), include_tensors=False).kv
            arch = kv.get("general.architecture")
            if arch is None:
                continue
            # Mixtral ships under general.architecture='llama' + an expert count;
            # index it as 'mixtral' so a parity test can select it distinctly from
            # a dense Llama (the loader still routes model_type=mixtral via the
            # same metadata signal).
            if arch == "llama" and int(kv.get("llama.expert_count") or 0) > 0:
                arch = "mixtral"
        except Exception:
            continue
        index.setdefault(arch, []).append(str(path))
    if not index:
        pytest.skip(f"no readable *.gguf found under {gguf_dir}")
    # Deterministic pick order: non-embedding/rerank names first, then smallest.
    # Generation-parity tests asking for e.g. "qwen3" must not land on an
    # embedding checkpoint that happens to sort or size first.
    for paths in index.values():
        paths.sort(key=lambda s: (
            ("embed" in Path(s).name.lower() or "rerank" in Path(s).name.lower()),
            Path(s).stat().st_size,
        ))
    return index


def require_arch(gguf_index: dict, arch: str) -> str:
    """Return one GGUF path for ``arch`` from the index, or skip the test."""
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR (have: {sorted(gguf_index)})")
    return paths[0]


@pytest.fixture(autouse=True)
def _isolated_xdg_data(tmp_path_factory, monkeypatch):
    # Chat autosave writes sessions under $XDG_DATA_HOME/gmlx/chats; keep
    # every test away from the user's real data dir.
    monkeypatch.setenv(
        "XDG_DATA_HOME", str(tmp_path_factory.mktemp("xdg-data"))
    )
