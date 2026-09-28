"""A left-padded Gemma-4 batch keeps each row's logits when a one-column
update follows a full sliding window and a wider update follows it.

The one-column update rotates each sliding layer's BatchRotatingKVCache.
Two shapes put a wider update after it: a text prefill whose chunk widths
run 1 and then 13, and the one-column final prefill forward and two decode
steps followed by a four-column verify block. In both the shorter row's
first real key is still inside the window, and each row matches
transformers run on that row alone. Tiny random-weight models on the CPU
stream.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.generate import ar

from gmlx.upstream.batch_rotating_mask_fix import (
    install_batch_rotating_mask_fix,
)

from test_gemma4_image_masks import (
    _HIDDEN,
    _TOL,
    _embeds,
    _models,
    _reference,
)

_W = 8


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    # The loader installs the stock mask fix with every model.
    assert install_batch_rotating_mask_fix()


def _rows(n1, n2, extra=0):
    """Each row's embeddings (its prompt plus ``extra`` columns) and the
    batch, the shorter row left-padded."""
    e1, e2 = _embeds(n1 + extra), _embeds(n2 + extra, seed=8)
    pad = np.zeros((1, n1 - n2, _HIDDEN), np.float32)
    return e1, e2, np.concatenate([e1, np.concatenate([pad, e2], 1)], 0)


def _batch(m, n1, n2, embeds, step):
    return ar.PromptProcessingBatch(
        model=m, uids=[0, 1],
        input_ids=[list(range(1, n1 + 1)), list(range(1, n2 + 1))],
        max_tokens=[4, 4], inputs_embeds=mx.array(embeds),
        prompt_kwargs={}, prefill_step_size=step)


def _row_reference(model_type, e):
    hf_mod, tc, hf, _lm, _target = _models(model_type, _W)
    return _reference(hf_mod, tc, hf, e, np.zeros(e.shape[:2], np.int64))


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_one_column_chunk_then_a_wider_one(model_type, served):
    _hf_mod, _tc, _hf, lm, _target = _models(model_type, _W)
    n1, n2 = 40, 25
    e1, e2, embeds = _rows(n1, n2)
    with mx.stream(mx.cpu):
        batch = _batch(lm, n1, n2, embeds, 8)
        widths = []
        for w in (8, 8, 1, 13, 8, 1):
            batch.prefill_step_size = w
            widths.append(batch.prompt_step())
        out = lm(None, inputs_embeds=batch._inputs_embeds,
                 cache=batch.prompt_cache)
        got = np.array(out.logits[:, -1, :].astype(mx.float32))
    assert widths == [8, 8, 1, 13, 8, 1]
    for row, e in enumerate((e1, e2)):
        ref = _row_reference(model_type, e)[-1]
        assert np.abs(got[row] - ref).max() <= _TOL, row


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_verify_block_after_decode_steps(model_type, path, served):
    _hf_mod, _tc, _hf, lm, target = _models(model_type, _W)
    m = lm if path == "plain" else target
    n1, n2 = 12, 5
    e1, e2, embeds = _rows(n1, n2, extra=6)
    with mx.stream(mx.cpu):
        batch = _batch(m, n1, n2, embeds[:, :n1], n1)
        widths = []
        while batch.needs_processing():
            widths.append(batch.prompt_step())
        got = []
        # The final prefill column, two decode steps, a verify block.
        for a, z in ((n1 - 1, n1), (n1, n1 + 1), (n1 + 1, n1 + 2),
                     (n1 + 2, n1 + 6)):
            out = m(None, inputs_embeds=mx.array(embeds[:, a:z]),
                    cache=batch.prompt_cache)
            got.append(np.array(out.logits.astype(mx.float32)))
        got = np.concatenate(got, 1)
    assert widths == [n1 - 1]
    for row, (e, n) in enumerate(((e1, n1), (e2, n2))):
        ref = _row_reference(model_type, e)[n - 1:]
        assert np.abs(got[row] - ref).max() <= _TOL, row
