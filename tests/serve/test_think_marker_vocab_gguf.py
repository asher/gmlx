"""The synthetic vocabs of tests/serve/test_think_marker_matrix.py against
the real vocabs.

For each matrix row whose vocab is under ``KQUANT_TEST_GGUF_DIR``, the
real tokenizer is built from the GGUF header (no weights load) with the
row's template, as text serve builds it. Then:

1. the structured-output gate on the real ids gives the row's first step,
   with thinking on and off;
2. each marker in the rendered prompts and the scripted reply encodes on
   the real vocab as it does on the synthetic one: whole (its own ids
   appear in the text's ids) or merged with a neighbour. A row that
   misses a merge fails here.

The Nemotron-3.5-Bolt-E4B row runs on the Gemma 4 vocab, which it shares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("llguidance")

import think_vocab as tv  # noqa: E402
import test_think_marker_matrix as mx_  # noqa: E402
from test_think_marker_matrix import serve as serve  # noqa: E402

from gmlx.gen.thinking_budget import rfind_seq  # noqa: E402

_GEMMA = ("gemma4", "gemma-4-12B-it-Q6_K.gguf")
_QWEN35 = ("qwen35", "Qwen3.5-9B-Q4_K_M.gguf")
_QWEN36 = ("qwen35", "Qwen_Qwen3.6-27B-Q4_K_S.gguf")

# Matrix row -> (architecture, GGUF file name).
VOCABS = {
    "fixtures/gemma4_12b_template.jinja": _GEMMA,
    "fixtures/gemma4_template.jinja": _GEMMA,
    "fixtures/nemotron_bolt_template.jinja": _GEMMA,
    "fixtures/qwen35_9b_template.jinja": _QWEN35,
    "fixtures/qwen35_template.jinja": _QWEN35,
    "fixtures/qwen36_template.jinja": _QWEN36,
    "fixtures/qwen38_template.jinja": _QWEN36,
    "fixtures/qwen3_template.jinja": ("qwen3", "Qwen3-0.6B-Q8_0.gguf"),
    "fixtures/smollm2_template.jinja": (
        "llama", "SmolLM2-135M-Instruct-Q8_0.gguf"),
    "fixtures/gptoss_template.jinja": ("gpt-oss", "gpt-oss-20b-MXFP4.gguf"),
    "fixtures/beam_template.jinja": (
        "beam", "Beam-1.0-501B-A23B-UD-IQ2_XXS-00001-of-00004.gguf"),
    "fixtures/flint_template.jinja": ("flint", "flint-int4-q4_1.gguf"),
}
# Rows with no vocab in the test folder.
NO_VOCAB = (
    "bundled/deepseek_v41", "fixtures/deepseek_v4_template.jinja",
    "fixtures/glm5_template.jinja", "fixtures/granite_switch_template.jinja",
    "fixtures/k2_horizon_template.jinja", "fixtures/kimi_k25_template.jinja",
    "fixtures/kimi_k3_template.jinja", "fixtures/minicpmv46_template.jinja",
    "fixtures/minicpmv47_template.jinja",
    "fixtures/minimax_m31_template.jinja", "fixtures/minimax_m31_turns.jinja",
    "fixtures/mistral3_template.jinja",
    "fixtures/mistral7b_v03_template.jinja",
    "fixtures/muse_spark_template.jinja", "fixtures/muse_template.jinja",
    "fixtures/nemotron_transcribe/chat_template.jinja",
    "fixtures/ricursive_template.jinja",
)


def test_every_matrix_row_names_its_vocab():
    assert set(VOCABS) | set(NO_VOCAB) == set(mx_._SERVED)
    assert not set(VOCABS) & set(NO_VOCAB)


@pytest.fixture
def real(gguf_index):
    """``real(key)``: the matrix row served on its real vocab."""
    from gmlx.load.config_synth import GGUF_ARCH_TO_MODEL_TYPE
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.tokenizer import finish_gguf_tokenizer, load_tokenizer_from_gguf

    def build(key):
        arch, name = VOCABS[key]
        path = next((p for p in gguf_index.get(arch, [])
                     if Path(p).name == name), None)
        if path is None:
            pytest.skip(f"{name} is not under KQUANT_TEST_GGUF_DIR")
        row = mx_.ROWS[key]
        raw = load_tokenizer_from_gguf(
            scan_gguf(path, include_tensors=False, array_limit=1 << 62).kv,
            arch,
            chat_template_override=mx_.TEMPLATES[key])
        finish_gguf_tokenizer(raw, GGUF_ARCH_TO_MODEL_TYPE.get(arch))
        proc = tv.text_processor(raw)
        rg = mx_._GEN.ResponseGenerator.__new__(mx_._GEN.ResponseGenerator)
        rg.tokenizer, rg.processor = proc.tokenizer, proc
        return row, mx_.Served(rg, proc.tokenizer, row.model_type)

    return build


_KEYS = [pytest.param(k, marks=pytest.mark.integration) for k in VOCABS]


@pytest.mark.parametrize("key", _KEYS)
def test_the_gate_on_the_real_vocab(serve, real, key):
    row, s = real(key)
    start = row.pair[0] if row.pair else None
    kwargs, kind = row.off or row.on
    mx_._first_step(s, mx_._request(s, kwargs=kwargs), kind, start)
    kwargs, kind = row.on
    mx_._first_step(s, mx_._request(s, kwargs=kwargs), kind, start)


def _whole(tok, text, marker) -> int:
    """How many of ``marker``'s places in ``text`` hold its own ids."""
    ids, seq = tv.ids(tok, text), tv.ids(tok, marker)
    n = 0
    while (i := rfind_seq(ids, seq)) >= 0:
        n += 1
        ids = ids[:i]
    return n


@pytest.mark.parametrize("key", _KEYS)
def test_markers_split_as_the_synthetic_vocab_splits_them(serve, real, key):
    row, s = real(key)
    synth = mx_._served(mx_.TEMPLATES[key], row, "text")
    texts = []
    for kwargs, _kind in filter(None, (row.on, row.off)):
        texts.append(mx_._request(s, kwargs=kwargs).prompt)
    texts.append(mx_._on_reply(row)[0])
    markers = [m for m in (row.pair or ()) if m]
    if row.pair == mx_._HARMONY:
        markers += ["<|channel|>final<|message|>", "<|start|>assistant"]
    checked = 0
    for text in texts:
        for marker in markers:
            if marker not in text:
                continue
            checked += 1
            assert _whole(s.tok, text, marker) == _whole(
                synth.tok, text, marker), (marker, text[-80:])
    assert checked or not row.pair
