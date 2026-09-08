"""Tripwire for docs/env-vars.md and docs/internals/debug-switches.md, in the
docs-to-code direction: every backticked GMLX_* or MLX_VLM_* name either file
documents must be read somewhere in gmlx/, so a renamed or removed variable
cannot linger in the docs. A must-document set built from the code and from
the names the docs carried when the file was created catches the reverse
drift for the variables that matter. CPU-only, no imports of gmlx runtime."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = [_ROOT / "docs" / "env-vars.md",
         _ROOT / "docs" / "internals" / "debug-switches.md"]
_NAME = re.compile(r"`((?:GMLX|MLX_VLM)_[A-Z0-9_]+)")

# Read sites: os.environ.get("X"), os.getenv("X"), os.environ["X"], "X" in
# os.environ, the gmlx.envflags helpers, and bare string literals (the
# config.py env dicts and any helper that takes the name as data).
_LITERAL = re.compile(r"[\"']((?:GMLX|MLX_VLM)_[A-Z0-9_]+)[\"']")

# Names that must be documented: the cli.md appendix rows at the time of the
# split, the residency overrides, and the config-key env names.
_MUST_DOCUMENT = {
    "GMLX_STREAM_GPU_TOKENS", "GMLX_STREAM_PREFETCH", "GMLX_DECODE_ARENA_GB",
    "GMLX_STREAM_KV_CTX", "GMLX_PREFILL_NOCACHE", "GMLX_DECODE_KV_RESERVE_GB",
    "GMLX_ARENA_STAGE_MAX_TOKENS", "GMLX_ARENA_SPLIT_MAX_TOKENS",
    "GMLX_DECODE_PRESSURE", "GMLX_GOVERNOR", "GMLX_GOV_KERNEL_FLOOR_GB",
    "GMLX_GOV_RESERVE_GB", "GMLX_GPU_KEEPWARM", "GMLX_GPU_RESIDENT",
    "GMLX_KEEPWARM_IDLE_S", "GMLX_DECODE_LOOKAHEAD",
    "GMLX_DECODE_LOOKAHEAD_PROBE", "GMLX_DECODE_RAM_FLOOR_GB",
    "GMLX_DECODE_PAGECACHE_GB", "GMLX_CACHE_LIMIT_GB", "GMLX_NATIVE_FP",
    "GMLX_ROPE_FACTORS", "GMLX_CASCADE_SDPA", "GMLX_CASCADE_MIN_P",
    "GMLX_SPARSE_ATTN", "GMLX_SPARSE_K", "GMLX_SPARSE_MIN_S",
    "GMLX_SPARSE_ARCHS", "GMLX_FUSED_GDN", "GMLX_QWEN_OWNED",
    "GMLX_GEMMA_OWNED", "GMLX_NO_FAMILY_DEFAULTS", "GMLX_DRAFT_BLOCK_SIZE",
    "GMLX_MTP_WIDTH_CAP", "GMLX_IGNORE_EOS", "GMLX_API_KEY",
    "MLX_VLM_RESIDENT_BUDGET_GB", "MLX_VLM_MAX_RESIDENT_MODELS",
    "MLX_VLM_PINNED_MODELS", "MLX_VLM_RESIDENT_TTL_DISABLE",
    "MLX_VLM_RESIDENT_TTL_TICK", "MLX_VLM_TOKEN_QUEUE_TIMEOUT",
}


def _documented() -> set:
    names = set()
    for doc in _DOCS:
        assert doc.is_file(), f"missing {doc}"
        names |= set(_NAME.findall(doc.read_text()))
    return names


def _read_in_code() -> set:
    names = set()
    for py in (_ROOT / "gmlx").rglob("*.py"):
        names |= set(_LITERAL.findall(py.read_text(encoding="utf-8")))
    return names


def test_every_documented_name_is_read_by_the_code():
    stale = _documented() - _read_in_code()
    assert not stale, f"documented but never read in gmlx/: {sorted(stale)}"


def test_must_document_set_is_documented():
    missing = _MUST_DOCUMENT - _documented()
    assert not missing, f"undocumented: {sorted(missing)}"


@pytest.mark.parametrize("doc", _DOCS, ids=lambda p: p.name)
def test_env_doc_is_ascii(doc):
    doc.read_bytes().decode("ascii")
