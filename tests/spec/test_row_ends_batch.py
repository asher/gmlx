"""The native head batch keeps row ends through uneven rounds
(gmlx/cache/row_ends.py).

The loop test drives _owned_decode_rounds_batch on a tiny random-weight
owned qwen3.5 target with a real QwenMTPDrafter head, four rows of
different lengths at width cap 4. The head's drafts are steered from each
row's greedy continuation so the rows accept 0, 1 and 2 tokens in turn,
and every row must decode as it does alone: its tokens, each target
verify's hidden states and each head logits call. The prefill and the
rounds run on the CPU stream in float32, so the batch and each row agree
to rounding. The gate tests check which batches arm.
"""

from __future__ import annotations

import dataclasses
import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.config")

import gmlx.spec.speculative as spec  # noqa: E402
from gmlx.cache import kv_cache as owned_cache  # noqa: E402
from gmlx.cache import row_ends  # noqa: E402
from gmlx.models.qwen35 import owned as qwen35_owned  # noqa: E402
from gmlx.spec.mtp_drafter import QwenMTPDrafter  # noqa: E402

_VOCAB = 128
_LENS = [9, 15, 5, 12]
_ROUNDS = 7


def _text_cfg():
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig

    return TextConfig(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_hidden_layers=4,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=_VOCAB,
        num_key_value_heads=2, max_position_embeddings=2048,
        tie_word_embeddings=True, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.25},
        full_attention_interval=4)


def _target():
    cfg = _text_cfg()
    top = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                          image_token_id=124, video_token_id=125,
                          vision_start_token_id=126)
    mx.random.seed(13)
    lm = qwen35_owned.language_model_class("qwen3_5")(cfg, top)
    mx.eval(lm.parameters())
    return lm, cfg


def _head(cfg):
    mod = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.config")
    mx.random.seed(11)
    head = QwenMTPDrafter(mod.Qwen3_5MTPConfig.from_dict({
        "model_type": "qwen3_5_mtp",
        "text_config": {f.name: getattr(cfg, f.name)
                        for f in dataclasses.fields(cfg)}}))
    mx.eval(head.parameters())
    return head


@pytest.fixture
def env(monkeypatch):
    for var in ("MLX_VLM_GGUF_SPEC_WIDTH_CAP", "GMLX_MTP_PREEMPT",
                "GMLX_MTP_RESUME", "GMLX_MTP_MIXED", "GMLX_ROW_ENDS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", str(len(_LENS)))
    monkeypatch.setattr(spec, "_width_cap_memo", ("", None))
    # The CPU keeps the rounds in exact float32 (as test_mtp_rotating_verify).
    monkeypatch.setattr(spec, "generation_stream", mx.cpu)
    return monkeypatch


def _alone(lm, row, n):
    """The row's greedy continuation, one token at a time."""
    c = lm.make_cache()
    out = lm(mx.array([row]), cache=c)
    toks = [int(mx.argmax(out.logits[0, -1]).item())]
    while len(toks) < n:
        out = lm(mx.array([[toks[-1]]]), cache=c)
        toks.append(int(mx.argmax(out.logits[0, -1]).item()))
    return toks


def _run(lm, cfg, rows, refs, base=0):
    """_ROUNDS rounds of the batch loop over ``rows`` (row j of the batch is
    row ``base + j`` of the test). The head drafts as it would; its tokens
    are then replaced so test row i accepts (i + round) % 3 of its two
    drafts. Returns each row's tokens, each target verify's hidden states,
    each head logits call, and the target and head row ends at each draft."""
    singles, bonus = [], []
    for row in rows:
        c = lm.make_cache()
        out = lm(mx.array([row]), cache=c)
        singles.append(c)
        bonus.append(int(mx.argmax(out.logits[0, -1]).item()))
    cache = [type(layer[0]).merge(list(layer)) for layer in zip(*singles)]
    fa = cache[cfg.full_attention_interval - 1]
    head = _head(cfg)
    draft, logits_fn = head.draft_block, head._logits
    rnd, verifies, head_logits, ends = [0], [], [], []

    def steered(last_bonus, hidden, kv, block_size, *args, **kwargs):
        out = draft(last_bonus, hidden, kv, block_size, *args, **kwargs)
        ends.append((fa._row_ends, head._cache[0]._row_ends))
        n = out.shape[1]
        steer = []
        for j, b in enumerate(np.array(last_bonus).tolist()):
            k = int(fa.offset[j].item()) - len(rows[j])
            assert refs[j][k] == b, (base + j, k)
            take = (base + j + rnd[0]) % 3
            want = refs[j][k + 1:k + 1 + n]
            steer.append([t if m < take else (t + 1) % _VOCAB
                          for m, t in enumerate(want)])
        rnd[0] += 1
        return mx.array(steer, dtype=out.dtype)

    def logits(h):
        out = logits_fn(h)
        head_logits.append(np.array(out.astype(mx.float32)))
        return out

    verify = spec._mtp_verify_target

    def record_verify(lm_, verify_input, prompt_cache, sampler, **kwargs):
        out = verify(lm_, verify_input, prompt_cache, sampler, **kwargs)
        verifies.append(np.array(out.hidden.astype(mx.float32)))
        return out

    head.draft_block, head._logits = steered, logits
    got = [[b] for b in bonus]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spec, "_mtp_verify_target", record_verify)
        gen = spec._owned_decode_rounds_batch(
            SimpleNamespace(language_model=lm), head, lm, cache, hidden=None,
            b=list(bonus), shared_kv=None, seed_tokens=None,
            emitted=[1] * len(rows), max_tokens=64, sampler=None,
            draft_block_size=None)
        for toks, meta in gen:
            for j, t in enumerate(toks):
                if t is not None:
                    got[j].append(t)
            if meta["round_pos"] == meta["round_len"] - 1 and rnd[0] >= _ROUNDS:
                gen.close()
                break
    return got, verifies, head_logits, ends


@pytest.mark.parametrize("keep", ["1", "0"])
def test_each_row_of_an_uneven_native_head_batch_decodes_as_alone(env, keep):
    env.setenv("GMLX_ROW_ENDS", keep)
    rng = np.random.default_rng(5)
    rows = [rng.integers(1, 120, n).tolist() for n in _LENS]
    with mx.stream(mx.cpu):
        lm, cfg = _target()
        refs = [_alone(lm, row, 40) for row in rows]
        got, verifies, head_logits, ends = _run(lm, cfg, rows, refs)
        alone = [_run(lm, cfg, [row], [ref], base=i)
                 for i, (row, ref) in enumerate(zip(rows, refs))]
    for i, toks in enumerate(got):
        assert len(toks) > _ROUNDS
        assert toks == refs[i][:len(toks)], i
        assert toks == alone[i][0][0], i
        # every target verify and every head step of the row, as alone
        assert len(alone[i][1]) == len(verifies)
        assert len(alone[i][2]) == len(head_logits)
        for batch, one in zip(verifies, alone[i][1]):
            np.testing.assert_allclose(batch[i], one[0], atol=1e-4, rtol=1e-4)
        for batch, one in zip(head_logits, alone[i][2]):
            np.testing.assert_allclose(batch[i], one[0], atol=1e-4, rtol=1e-4)
    uneven = [e for e in ends if e[0] is not None]
    if keep == "1":
        # the rounds after the capture ran on uneven rows, target and head
        assert uneven and any(h is not None for _, h in uneven)
    else:
        assert not uneven


def test_an_emptied_uneven_batch_adopts_a_row_as_alone(env):
    """Two rows finish together in a round they leave ending apart (takes 1
    and 2) while a third row's admission is queued. The emptied batch adopts
    it: the row's tokens are its greedy continuation and each verify's
    hidden states equal the same verify on the row alone."""
    rng = np.random.default_rng(7)
    rows = [rng.integers(1, 120, n).tolist() for n in (9, 15, 11)]
    fa_idx = 3
    with mx.stream(mx.cpu):
        lm, cfg = _target()
        fa_idx = cfg.full_attention_interval - 1
        refs = [_alone(lm, row, 40) for row in rows]
        singles, bonus = [], []
        for row in rows[:2]:
            c = lm.make_cache()
            lm(mx.array([row]), cache=c)
            singles.append(c)
        bonus = [refs[0][0], refs[1][0]]
        cache = [type(layer[0]).merge(list(layer)) for layer in zip(*singles)]
        joiner = lm.make_cache()
        jv = spec._mtp_verify_target(lm, mx.array([rows[2]]), joiner, None)
        assert int(jv.target_tokens[0, -1].item()) == refs[2][0]
        head = _head(cfg)
        draft = head.draft_block
        live, rnd, verifies, stale = [0, 1], [0], [], []

        def steered(last_bonus, hidden, kv, block_size, *args, **kwargs):
            out = draft(last_bonus, hidden, kv, block_size, *args, **kwargs)
            fa = cache[fa_idx]
            steer = []
            for j, b in enumerate(np.array(last_bonus).tolist()):
                i = live[j]
                k = int(fa.offset[j].item()) - len(rows[i])
                assert refs[i][k] == b, (i, k)
                take = (i + rnd[0]) % 3
                want = refs[i][k + 1:k + 1 + out.shape[1]]
                steer.append([t if m < take else (t + 1) % _VOCAB
                              for m, t in enumerate(want)])
            rnd[0] += 1
            return mx.array(steer, dtype=out.dtype)

        verify = spec._mtp_verify_target

        def record_verify(lm_, verify_input, prompt_cache, sampler, **kwargs):
            k = [int(o) for o in prompt_cache[fa_idx].offset.tolist()]
            out = verify(lm_, verify_input, prompt_cache, sampler, **kwargs)
            verifies.append((list(live), k, np.array(verify_input).tolist(),
                             np.array(out.hidden.astype(mx.float32))))
            return out

        empty_rows = spec._filter_batch_rows_empty

        def spy_empty(prompt_cache):
            stale.append([getattr(c, "_row_ends", None) for c in prompt_cache])
            empty_rows(prompt_cache)

        counts, ends_at = {0: 1, 1: 1}, {0: 5, 1: 7}

        def stop(orig, tok):
            # the first two rows end together with the second draft round,
            # which leaves them ending 2 apart
            if orig in counts:
                counts[orig] += 1
                return counts[orig] >= ends_at[orig]
            return False

        head.draft_block = steered
        model = SimpleNamespace(language_model=lm)
        got = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(spec, "_mtp_verify_target", record_verify)
            mp.setattr(spec, "_filter_batch_rows_empty", spy_empty)
            gen = spec._owned_decode_rounds_batch(
                model, head, lm, cache, hidden=None, b=list(bonus),
                shared_kv=None, seed_tokens=None, emitted=[1, 1],
                max_tokens=64, sampler=None, draft_block_size=None,
                stop_check=stop)
            for toks, meta in gen:
                if len(toks) > 2 and toks[2] is not None:
                    got.append(toks[2])
                last = meta["round_pos"] == meta["round_len"] - 1
                if rnd[0] == 2 and not getattr(model, "_generator_injections", None) \
                        and len(toks) == 2:
                    model._generator_injections = [{
                        "uids": [None], "prompt_cache": joiner,
                        "prompt_tokens": mx.array([rows[2]]),
                        "hidden": jv.hidden, "first_tokens": jv.target_tokens[:, -1],
                        "first_tokens_list": [refs[2][0]], "max_tokens": [64]}]
                    live[:] = [2]
                if last and rnd[0] >= 7:
                    gen.close()
                    break
        # the batch emptied after an uneven round, then adopted the row
        assert stale and [e for e in stale[0] if e is not None] == [[19, 21]]
        adopted = [v for v in verifies if v[0] == [2]]
        assert len(adopted) >= 3
        assert [refs[2][0]] + got == refs[2][:len(got) + 1]
        for _, (k,), x, hidden in adopted:
            prefix = rows[2] + refs[2][:k - len(rows[2])]
            c = lm.make_cache()
            lm(mx.array([prefix]), cache=c)
            want = verify(lm, mx.array(x), c, None).hidden
            np.testing.assert_allclose(hidden, np.array(want.astype(mx.float32)),
                                       atol=1e-4, rtol=1e-4)


def test_the_native_head_on_a_dense_target_arms(env):
    lm, cfg = _target()
    head = _head(cfg)
    assert row_ends.row_ends_active(head, lm, mixed=False)


def test_mixed_and_switched_off_batches_never_arm(env):
    lm, cfg = _target()
    head = _head(cfg)
    assert not row_ends.row_ends_active(head, lm, mixed=True)
    env.setenv("GMLX_ROW_ENDS", "0")
    assert not row_ends.row_ends_active(head, lm, mixed=False)


def test_other_targets_and_drafters_never_arm(env):
    from gmlx.models.qwen35.language import LanguageModel

    lm, cfg = _target()
    head = _head(cfg)
    moe = qwen35_owned.language_model_class("qwen3_5_moe")
    assert not getattr(moe, "row_ends_rollback", False)
    stock = LanguageModel.__new__(LanguageModel)
    assert not row_ends.row_ends_active(head, stock, mixed=False)
    # a drafter without the flag (DFlash2, the plain batch's fakes)
    assert not row_ends.row_ends_active(SimpleNamespace(), lm, mixed=False)


def test_a_trip_right_justifies_and_disarms():
    from gmlx.models.qwen35.language import rollback_attention_caches

    def batch(armed):
        c = owned_cache.BatchKVCache([3, 0, 1])
        c._row_ends_armed = armed
        k = mx.random.normal((3, 2, 12, 64), key=mx.random.key(1))
        c.update_and_fetch(k, k)
        v = mx.random.normal((3, 2, 3, 64), key=mx.random.key(2))
        c.update_and_fetch(v, v)
        rollback_attention_caches([c], [2, 0, 1], 3)
        return c

    armed, plain = batch(True), batch(False)
    assert armed._row_ends is not None
    row_ends.disarm([owned_cache.CacheList(armed)])
    assert armed._row_ends is None and not armed._row_ends_armed
    assert armed.left_padding.tolist() == plain.left_padding.tolist()
    n = plain._idx
    for i, p in enumerate(plain.left_padding.tolist()):
        np.testing.assert_array_equal(np.asarray(armed.keys[i, :, p:n]),
                                      np.asarray(plain.keys[i, :, p:n]))
