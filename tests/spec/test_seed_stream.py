#!/usr/bin/env python3
"""MTP drafter seed streaming: chunked prefill-time teacher-force parity.

The stall this feature removes: the deferred whole-prompt seed used to run
after the first token (seconds at depth). Streaming teacher-forces each
prefill chunk into a request-scoped cache; the owned round adopts it via
restore_kv and seeds only the residual.

Unit layer: a fake head keeps the base-class seed_chunk / seed_finish /
prefill_from_target_hidden logic (the code under test) on real KVCache
mechanics, with a chunk-invariant but offset- and token-sensitive forward,
so any offset, shift-by-one, or adoption bug changes the numbers. Real-head
numerics are covered by the gguf-gated integration suite
(test_full_prompt_prefill.py).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import gmlx.spec.engine as engine  # noqa: E402
from gmlx.spec.mtp_drafter import QwenMTPDrafter  # noqa: E402

D = 8
VOCAB = 32


class _StubCfg:
    block_size = 3
    text_config = None
    model_type = None


class FakeHead(QwenMTPDrafter):
    """Base-class seeding logic on a deterministic toy forward.

    out[s] = mean over the cache's values up to position s -- invariant to
    how the span is chunked, but sensitive to the cache offset (a chunk
    seeded at the wrong offset changes every later mean) and to the token
    pairing (tokens enter the values).
    """

    supports_kv_sidecar = False  # the adoption seam must not depend on this

    def __init__(self):
        nn.Module.__init__(self)
        self.config = _StubCfg()
        self._native_block_size = 3
        self.layers = [None]
        self._postnorm_feed = False
        self._input_embed = None
        self._input_embed_scale = 1.0
        self._lm_head_fn = None
        self._cache = []
        self._seed_token = None
        self._seed_hidden = None
        self._round_appended = 0
        self.accept_lens = []
        self.draft_lens = []

    def bind(self, target_model):
        self._input_embed = lambda t: t.astype(mx.float32)[..., None]
        self._lm_head_fn = lambda h: mx.concatenate(
            [h, mx.zeros(h.shape[:-1] + (VOCAB - D,))], axis=-1)
        return self

    def norm(self, x):
        return x

    def _forward(self, tokens, hidden, cache=None):
        caches = self._cache if cache is None else cache
        c = caches[0]
        v = hidden + tokens.astype(mx.float32)[..., None]
        k = v * 0.5
        _, V = c.update_and_fetch(k[:, None], v[:, None])
        S = int(tokens.shape[1])
        off = int(V.shape[2]) - S
        outs = [mx.mean(V[:, 0, : off + s + 1, :], axis=1) for s in range(S)]
        return mx.stack(outs, axis=1)


def _prompt(n, seed=7):
    return mx.array([[(seed + 3 * i) % VOCAB for i in range(n)]],
                    dtype=mx.int32)


def _hidden(n, seed=1):
    return mx.arange(n * D, dtype=mx.float32).reshape(1, n, D) * 0.01 + seed


class _Target:
    pass


def _one_shot(ids, hid, bonus):
    """Reference: today's deferred whole-prompt seed."""
    d = FakeHead()
    d.reset(_Target())
    d.prefill_from_target_hidden(ids, hid, bonus, None, greedy=True)
    return d


def _streamed(ids, hid, bonus, splits):
    """Stream spans (engine prompt_step twin), adopt, seed the residual."""
    n = int(ids.shape[1])
    d = FakeHead()
    d.bind(_Target())
    seed_kv = d.make_cache()
    c0 = 0
    for take in splits:
        d.seed_chunk(ids[:, c0 + 1:c0 + take + 1], hid[:, c0:c0 + take],
                     seed_kv)
        c0 += take
    assert c0 < n, "splits must leave a residual"
    # Owned-round order: reset (kills the drafter's own cache), then the
    # direct adoption, then the residual seed over the trailing hidden.
    d.reset(_Target())
    assert int(seed_kv[0].offset) == c0
    d.restore_kv(seed_kv)
    d.prefill_from_target_hidden(ids, hid[:, c0:], bonus, None, greedy=True)
    return d, c0


def _state(d):
    ks, vs = d._cache[0].state[0], d._cache[0].state[1]
    off = int(d._cache[0].offset)
    return ks[:, :, :off], vs[:, :, :off]


N = 13
SPLITS = ([4, 4, 4], [1, 2, 3, 4], [12], [5, 5])


class TestStreamedParity:

    @pytest.mark.parametrize("splits", SPLITS)
    def test_kv_and_seed_match_one_shot(self, splits):
        ids, hid, bonus = _prompt(N), _hidden(N), 21
        ref = _one_shot(ids, hid, bonus)
        got, _ = _streamed(ids, hid, bonus, splits)
        for a, b in zip(_state(ref), _state(got)):
            assert mx.allclose(a, b, atol=1e-6), "head KV diverged"
        assert int(ref._seed_token.item()) == int(got._seed_token.item())
        assert mx.allclose(ref._seed_hidden, got._seed_hidden, atol=1e-6)

    def test_offset_tracks_streamed_len(self):
        ids, hid = _prompt(N), _hidden(N)
        d = FakeHead()
        d.bind(_Target())
        kv = d.make_cache()
        d.seed_chunk(ids[:, 1:6], hid[:, 0:5], kv)
        assert int(kv[0].offset) == 5
        d.seed_chunk(ids[:, 6:10], hid[:, 5:9], kv)
        assert int(kv[0].offset) == 9

    def test_seed_chunk_never_samples(self):
        calls = []
        sampler = lambda logits: (calls.append(1), mx.argmax(logits, -1))[1]
        ids, hid = _prompt(N), _hidden(N)
        d = FakeHead()
        d.bind(_Target())
        kv = d.make_cache()
        d.seed_chunk(ids[:, 1:9], hid[:, 0:8], kv)
        assert not calls
        d.reset(_Target())
        d.restore_kv(kv)
        d.prefill_from_target_hidden(ids, hid[:, 8:], 21, sampler)
        assert len(calls) == 1, "exactly one seed-pick deposit"

    def test_mid_stop_partial_adopt_beats_cold_reseed(self):
        """The partial cache MUST be adopted: residual-at-offset matches the
        one-shot; a cold offset-0 reseed of the same residual does not."""
        ids, hid, bonus = _prompt(N), _hidden(N), 21
        ref = _one_shot(ids, hid, bonus)
        got, c0 = _streamed(ids, hid, bonus, [4, 3])  # stop after 7 of 13
        for a, b in zip(_state(ref), _state(got)):
            assert mx.allclose(a, b, atol=1e-6)
        cold = FakeHead()
        cold.reset(_Target())
        cold.prefill_from_target_hidden(ids, hid[:, c0:], bonus, None,
                                        greedy=True)
        assert int(cold._cache[0].offset) != int(ref._cache[0].offset)

    def test_untouched_own_cache_during_streaming(self):
        """seed_chunk writes only the passed cache -- a live round's _cache
        (another request) must never grow."""
        ids, hid = _prompt(N), _hidden(N)
        d = FakeHead()
        d.reset(_Target())
        live = d._cache
        kv = d.make_cache()
        d.seed_chunk(ids[:, 1:9], hid[:, 0:8], kv)
        assert d._cache is live
        assert int(getattr(live[0], "offset", 0)) == 0


class _StubBatch:
    """Just the attrs _mtp_seed_stream_init reads."""

    def __init__(self, drafter, b=1, chunk_hiddens=(), l1=0, processed=0,
                 warm=False, cache_attr=None):
        self.draft_model = drafter
        self.model = _Target()
        self._input_ids = mx.zeros((b, 16), dtype=mx.int32)
        self._mtp_upstream_warm = warm
        self._mtp_chunk_hiddens = list(chunk_hiddens)
        self._mtp_l1_prefix_len = l1
        self._processed_prompt_columns = processed
        self.prompt_cache = [type("C", (), {})()]
        if cache_attr:
            for k, v in cache_attr.items():
                setattr(self.prompt_cache[0], k, v)


class TestEligibility:

    def _armed(self, batch):
        engine._mtp_seed_stream_init(batch)
        return batch._mtp_seed_ctx is not None

    def test_eligible_cold_prefill_arms(self):
        assert self._armed(_StubBatch(FakeHead()))

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setattr(engine, "_SEED_STREAM_DISABLED", True)
        assert not self._armed(_StubBatch(FakeHead()))

    def test_window_limited_head_defers(self):
        d = FakeHead()
        d.hidden_capture_limit = 128  # deepseek_v4 / dspark / windowed q4e
        assert not self._armed(_StubBatch(d))

    def test_missing_seed_chunk_defers(self):
        class _NoStream:
            hidden_capture_limit = None
        assert not self._armed(_StubBatch(_NoStream()))

    def test_b_gt_1_defers(self):
        assert not self._armed(_StubBatch(FakeHead(), b=2))

    def test_l0_hit_defers(self):
        h = _hidden(4)
        assert not self._armed(_StubBatch(FakeHead(), chunk_hiddens=[h]))

    def test_l1_prefix_defers(self):
        assert not self._armed(_StubBatch(FakeHead(), l1=64, processed=64))

    def test_upstream_warm_defers(self):
        assert not self._armed(_StubBatch(FakeHead(), warm=True))

    def test_warm_sidecar_defers(self):
        assert not self._armed(_StubBatch(
            FakeHead(), cache_attr={"_kq_apc_drafter_warm": [object()]}))

    def test_armed_ctx_rides_prompt_cache(self):
        b = _StubBatch(FakeHead())
        engine._mtp_seed_stream_init(b)
        assert b.prompt_cache[0]._kq_seed_stream is b._mtp_seed_ctx
        assert b._mtp_seed_ctx["len"] == 0
        assert b._mtp_seed_ctx["active"]
