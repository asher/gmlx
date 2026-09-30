"""Batched MTP rounds on a Gemma-4 target keep a left-padded row's verify
block equal to its single-row run after a one-column forward rotates the
sliding caches.

The owned batched loop enters from a preempt rebuild without hidden
states, so its first round is a one-column capture forward. With one row
past the sliding window, that forward rotates each sliding layer's
BatchRotatingKVCache, and the next round's verify block is wider. The
shorter, left-padded row still has its first real key inside the window.
The first test stops after that verify block, before any rollback runs. The
second runs on through a rollback, an admission and a one-token step. The
third runs a lone row through a rollback and then a forced close of one id,
a width-1 verify. A tiny random-weight owned target, with the prefill and
the rounds on the CPU stream, and a drafter that proposes two fixed offsets
of the bonus token.
"""

import functools
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from gmlx._vendor.mlx_vlm.models import cache as vlm_cache
from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig

import gmlx.models.gemma4.sync as gemma4_sync
import gmlx.spec.speculative as spec
from gmlx.models.gemma4.owned import OwnedGemma4LanguageModel

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

    def inject_rows(self, *args, **kwargs):
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


def test_a_gated_round_after_an_admission_and_a_rejected_draft(
        target, monkeypatch):
    """Two rows speculate past the window and reject both drafts, and the
    rollback trims them. A third row joins, the width gate trips, and the
    next round is a one-token step for all three rows. Each row's logits in
    that step must match a single-row forward over the row's history."""
    lm, cls = target, type(target)
    rng = np.random.default_rng(3)
    rows = [rng.integers(1, _VOCAB, 30).tolist(),
            rng.integers(1, _VOCAB, 6).tolist()]
    joining = rng.integers(1, _VOCAB, 5).tolist()
    call, trim = cls.__call__, vlm_cache.BatchRotatingKVCache.trim
    gated, trims = [], []

    def record_call(self, inputs, *args, **kwargs):
        out = call(self, inputs, *args, **kwargs)
        if inputs.ndim == 2 and inputs.shape == (3, 1):
            gated.append((np.array(inputs)[:, 0].tolist(),
                          np.array(out.logits[:, -1].astype(mx.float32))))
        return out

    @functools.wraps(trim)
    def record_trim(self, n):
        trims.append(n)
        return trim(self, n)

    def truth(history):
        out = call(lm, mx.array([history]), cache=lm.make_cache())
        return np.array(out.logits[0, -1].astype(mx.float32))

    monkeypatch.setattr(cls, "__call__", record_call)
    monkeypatch.setattr(vlm_cache.BatchRotatingKVCache, "trim", record_trim)
    with mx.stream(mx.cpu):
        singles, bonus = [], []
        for row in rows + [joining]:
            c = lm.make_cache()
            out = call(lm, mx.array([row]), cache=c)
            singles.append(c)
            bonus.append(int(mx.argmax(out.logits[0, -1]).item()))
        cache = [type(layer[0]).merge(list(layer))
                 for layer in zip(*singles[:2])]
        model = SimpleNamespace()
        history = [rows[0] + [bonus[0]], rows[1] + [bonus[1]]]
        gen = spec._owned_decode_rounds_batch(
            model, _Drafter(), lm, cache, hidden=None, b=bonus[:2],
            shared_kv=None, seed_tokens=None, emitted=[1, 1],
            max_tokens=40, sampler=None, draft_block_size=None)
        rounds, at_admission = 0, None
        for tokens, meta in gen:
            for j, t in enumerate(tokens[:2]):
                if t is not None:
                    history[j].append(int(t))
            if meta["round_pos"] == meta["round_len"] - 1:
                rounds += 1
                if rounds == 2:
                    at_admission = [list(h) for h in history]
                    model._generator_injections = [{
                        "uids": [None], "prompt_cache": singles[2],
                        "prompt_tokens": mx.array([joining]), "hidden": None,
                        "first_tokens": mx.array([bonus[2]]),
                        "first_tokens_list": [bonus[2]], "max_tokens": [40]}]
            if gated:
                gen.close()
                break
        assert at_admission is not None and gated
        # The drafts were rejected before the admission: the rollback trimmed
        # them from the sliding caches.
        assert any(n > 0 for n in trims), trims
        histories = at_admission + [joining + [bonus[2]]]
        inputs, logits = gated[0]
        assert inputs == [h[-1] for h in histories]
        worst = []
        for j, h in enumerate(histories):
            want = truth(h)
            d = float(np.abs(logits[j] - want).max())
            worst.append(d / float(np.abs(want).max()))
    # Each row agrees with its own forward to about 2e-6 of the logit scale;
    # the rejected drafts left in the ring move them by about the scale.
    assert all(r <= 1e-4 for r in worst), worst


class _BudgetClose:
    """A thinking budget that closes with one id once a rollback has
    trimmed the sliding caches."""

    budget = None

    def __init__(self, trims):
        self.trims, self.fired = trims, False

    def observe(self, tok):
        pass

    def take_forced(self):
        if self.fired or not any(self.trims):
            return None
        self.fired = True
        return [7]


def test_a_forced_one_id_round_after_a_rejected_draft(target, monkeypatch):
    """A lone row past the window rejects both drafts, and the rollback
    trims them. Its thinking budget then closes with one id, which runs as
    a width-1 verify of the bonus token. That verify must match a
    single-row forward over the row's history."""
    lm = target
    row = np.random.default_rng(3).integers(1, _VOCAB, 30).tolist()
    verify, trim = spec._mtp_verify_target, vlm_cache.BatchRotatingKVCache.trim
    history, trims, forced = [], [], []

    def record_verify(lm_, verify_input, prompt_cache, sampler, **kwargs):
        out = verify(lm_, verify_input, prompt_cache, sampler, **kwargs)
        # The capture round is width 1 too, but runs before any rollback.
        if verify_input.shape[1] == 1 and any(trims):
            forced.append((int(verify_input[0, 0].item()), list(history),
                           np.array(out.hidden[0, -1].astype(mx.float32)),
                           kwargs))
        return out

    @functools.wraps(trim)
    def record_trim(self, n):
        trims.append(n)
        return trim(self, n)

    monkeypatch.setattr(spec, "_mtp_verify_target", record_verify)
    monkeypatch.setattr(vlm_cache.BatchRotatingKVCache, "trim", record_trim)
    with mx.stream(mx.cpu):
        single = lm.make_cache()
        out = lm(mx.array([row]), cache=single)
        bonus = int(mx.argmax(out.logits[0, -1]).item())
        cache = [type(c).merge([c]) for c in single]
        history.extend(row + [bonus])
        gen = spec._owned_decode_rounds_batch(
            SimpleNamespace(), _Drafter(), lm, cache, hidden=None, b=[bonus],
            shared_kv=None, seed_tokens=None, emitted=[1], max_tokens=40,
            sampler=None, draft_block_size=None,
            thinking_hook=_BudgetClose(trims))
        for tokens, _meta in gen:
            if tokens[0] is not None:
                history.append(int(tokens[0]))
            if forced:
                gen.close()
                break
        # One forced round ran, after a rollback trimmed rejected drafts.
        assert len(forced) == 1 and any(n > 0 for n in trims), (forced, trims)
        token, seen, got, kwargs = forced[0]
        assert token == seen[-1]
        truth = verify(lm, mx.array([seen]), lm.make_cache(), None, **kwargs)
        want = np.array(truth.hidden[0, -1].astype(mx.float32))
    ratio = float(np.abs(got - want).max()) / float(np.abs(want).max())
    # The verify agrees with the fresh forward to about 2e-6 of the hidden
    # scale; the rejected drafts left in the ring move it by far more.
    assert ratio <= 1e-4, ratio


def test_owned_rotating_caches_take_the_rotating_branch_of_the_rollback():
    """The rollback transaction binds the rotating classes from the cache
    module at import. Both names are gmlx's own classes, so an owned
    rotating cache gets the window-replay transaction, not the cursor
    trim, and a rejected block leaves only the accepted keys."""
    from gmlx._vendor.mlx_vlm.speculative import cache_state

    import gmlx.cache.kv_cache as owned

    assert cache_state.RotatingKVCache is owned.RotatingKVCache
    assert cache_state.BatchRotatingKVCache is owned.BatchRotatingKVCache
    assert vlm_cache.RotatingKVCache is owned.RotatingKVCache
    single = owned.RotatingKVCache(max_size=8)
    batch = owned.BatchRotatingKVCache(8, [0, 0])

    def keys(cols, rows):
        k = mx.array(cols, dtype=mx.float32)[None, None, :, None] + 1
        return mx.broadcast_to(k, (rows, 1, len(cols), 2))

    for cache, rows in ((single, 1), (batch, 2)):
        cache.update_and_fetch(keys(list(range(10)), rows),
                               keys(list(range(10)), rows))
    tx = cache_state.start_speculative_cache([single, batch], 3)
    assert set(tx._rotating) == {id(single), id(batch)}
    assert not tx._positions
    for cache, rows in ((single, 1), (batch, 2)):
        cache.update_and_fetch(keys([10, 11, 12], rows),
                               keys([10, 11, 12], rows))
    tx.commit(1)
    for cache in (single, batch):
        kept = cache._temporal_order(cache.keys) if cache is single \
            else cache.keys
        cols = sorted(int(x) - 1 for x in kept[0, 0, :, 0].tolist())
        assert 10 in cols and 11 not in cols and 12 not in cols, (
            type(cache).__name__, cols)
