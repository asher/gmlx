"""Batched MTP rounds on a Gemma-4 target keep a left-padded row's verify
block equal to its single-row run after a one-column forward rotates the
sliding caches.

The owned batched loop enters from a preempt rebuild without hidden
states, so its first round is a one-column capture forward. With one row
past the sliding window, that forward rotates each sliding layer's
BatchRotatingKVCache, and the next round's verify block is wider. The
shorter, left-padded row still has its first real key inside the window.
The test stops after that verify block, before any rollback runs. A tiny
random-weight owned target, with the prefill and the rounds on the CPU
stream, and a drafter that proposes two fixed offsets of the bonus token.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.gemma4.config import TextConfig

import gmlx.models.gemma4.sync as gemma4_sync
import gmlx.spec.speculative as spec
from gmlx.models.gemma4.owned import OwnedGemma4LanguageModel
from gmlx.upstream.batch_rotating_mask_fix import (
    install_batch_rotating_mask_fix,
)

_W, _VOCAB = 16, 128


class _Drafter:
    uses_shared_kv = True
    mtp_width_cap = 2
    mtp_width_limit = 0

    def __init__(self):
        self.config = SimpleNamespace(block_size=3)
        self.accept_lens, self.draft_lens = [], []

    def reset(self, model, left_padding=None):
        return []

    def set_shared_kv(self, *args, **kwargs):
        pass

    def prefill_from_target_hidden(self, *args, **kwargs):
        pass

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kwargs):
        return mx.stack([(b + 61) % _VOCAB, (b + 83) % _VOCAB],
                        axis=1).astype(dtype)


@pytest.fixture
def target(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_PREEMPT", "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spec, "_width_cap_memo", ("", None))
    # The rounds run on the engine's stream: the CPU keeps them in exact
    # float32, so near-tie picks agree between the batch and each row.
    monkeypatch.setattr(spec, "generation_stream", mx.cpu)
    gemma4_sync.install_gemma4_nosync()
    # The loader installs the stock mask fix with every model.
    assert install_batch_rotating_mask_fix()
    cfg = TextConfig(
        model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
        intermediate_size=128, num_attention_heads=4, head_dim=16,
        global_head_dim=32, rms_norm_eps=1e-6, vocab_size=_VOCAB,
        vocab_size_per_layer_input=_VOCAB, num_key_value_heads=2,
        num_kv_shared_layers=2, hidden_size_per_layer_input=0,
        sliding_window=_W, sliding_window_pattern=3,
        tie_word_embeddings=True)
    mx.random.seed(13)
    lm = OwnedGemma4LanguageModel(cfg)
    mx.eval(lm.parameters())
    return lm


def _rounds(lm, rows, rounds=2):
    """Each verify block's inputs and target hidden states, and the
    (width, rotated) of every sliding-cache mask, over the first
    ``rounds`` rounds from the preempt entry."""
    with pytest.MonkeyPatch.context() as mp:
        return _record(lm, rows, rounds, mp)


def _record(lm, rows, rounds, monkeypatch):
    verifies, masks = [], []
    verify = spec._mtp_verify_target

    def record_verify(lm_, verify_input, prompt_cache, sampler, **kwargs):
        out = verify(lm_, verify_input, prompt_cache, sampler, **kwargs)
        verifies.append((np.array(verify_input).tolist(),
                         np.array(out.hidden.astype(mx.float32))))
        return out

    make_mask = vlm_cache.BatchRotatingKVCache.make_mask

    def record_mask(self, N, window_size=None, return_array=False):
        masks.append((N, bool(self.rotated)))
        return make_mask(self, N, window_size=window_size,
                         return_array=return_array)

    monkeypatch.setattr(spec, "_mtp_verify_target", record_verify)
    monkeypatch.setattr(vlm_cache.BatchRotatingKVCache, "make_mask",
                        record_mask)
    singles, bonus = [], []
    for row in rows:
        c = lm.make_cache()
        out = lm(mx.array([row]), cache=c)
        singles.append(c)
        bonus.append(int(mx.argmax(out.logits[0, -1]).item()))
    cache = [type(layer[0]).merge(list(layer)) for layer in zip(*singles)]
    gen = spec._owned_decode_rounds_batch(
        SimpleNamespace(), _Drafter(), lm, cache, hidden=None, b=bonus,
        shared_kv=None, seed_tokens=None, emitted=[1] * len(rows),
        max_tokens=40, sampler=None, draft_block_size=None)
    for n, _step in enumerate(gen):
        if n >= rounds - 1:
            gen.close()
            break
    return verifies, masks


def test_a_verify_block_after_a_capture_forward(target):
    rng = np.random.default_rng(3)
    rows = [rng.integers(1, _VOCAB, 30).tolist(),
            rng.integers(1, _VOCAB, 6).tolist()]
    with mx.stream(mx.cpu):
        batched, masks = _rounds(target, rows)
        alone = [_rounds(target, [row])[0] for row in rows]
    # A verify block ran on sliding caches a one-column forward rotated.
    assert masks[:2] == [(1, False), (3, True)]
    assert [len(v[0][0]) for v in batched] == [1, 3]
    worst = []
    for k, (inputs, hidden) in enumerate(batched):
        for row in range(2):
            (one_inputs, one_hidden) = alone[row][k]
            assert one_inputs[0] == inputs[row]
            scale = float(np.abs(one_hidden).max())
            d = float(np.abs(hidden[row] - one_hidden[0]).max())
            worst.append((k, row, d / scale))
    # The batch and each row agree to about 2e-6 of the hidden scale; the
    # lost key moves the row by more than half of it.
    assert all(r <= 1e-4 for _k, _row, r in worst), worst
