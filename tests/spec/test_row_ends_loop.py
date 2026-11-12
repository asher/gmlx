"""Row ends in the native head batch loop (gmlx/cache/row_ends.py).

A tiny owned Qwen3.5 target with its random-weight MTP head serves four
rows of different lengths at width 4. Drafts are steered so the rows
accept 0, 1 and 2 tokens in the same round: each round leaves the rows
ending apart, and the next verify reads each row's keys up to its own end.
Under greedy verify the drafts decide only how many tokens a round keeps,
so every row's tokens equal the same row decoded alone. The gate tests
pin which batches arm: the mixed batch, a gated batch, DFlash2, a MoE head
and every target but the owned dense one never do, and a trip to plain
decode right-justifies the rows once."""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen3_5.language")

import gmlx.models.qwen35.attn as qwen35_attn
import gmlx.models.qwen35.language as qwen35_language
import gmlx.spec.speculative as spec
from gmlx.cache import kv_cache as owned
from gmlx.cache import row_ends
from gmlx.cache.kvarn_sdpa import kvarn_row_ends_ok
from gmlx.spec.admission import install_continuous_batch_admission
from gmlx.spec.engine import install_owned_spec_engine

from test_apc_causal_media import _manager, _mtp_model, _request, serve  # noqa: F401
from test_mtp_width_cap import (
    _FakeCache, _FakeLM, _StrictDrafter, _clean_width_cap_env,  # noqa: F401
    _drive, _queue_injection)

import importlib

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

_needs_ends = pytest.mark.skipif(
    not kvarn_row_ends_ok(), reason="mlx-kquant takes no per-row ends")
_metal = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the owned qwen3_5 target's GDN kernels are Metal-only")

_LENS = (20, 27, 33, 41)
# Accepts per row in each of the first three rounds. Every round but the
# last leaves the rows ending apart.
_SCHEDULE = ([2, 0, 1, 2], [0, 2, 2, 1], [1, 1, 0, 2])
_MAX = 16


def _prompts():
    rng = np.random.default_rng(3)
    return [rng.integers(1, 120, n).tolist() for n in _LENS]


def _run(model, rows, head, max_tokens=_MAX):
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria

    model.language_model._spec_prefix_cache = None
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(model.language_model, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=max_tokens, apc_manager=_manager(),
                            draft_model=head, draft_kind="mtp",
                            greedy_sampling=True)
    uids = gen.insert([r[0] for r in rows], [max_tokens] * len(rows),
                      prompt_kwargs=[r[1] for r in rows])
    out = {u: [] for u in uids}
    while gen.has_work:
        _p, resp = gen.next()
        for r in resp:
            if r.uid in out:
                out[r.uid].append(int(r.token))
    gen.close()
    return [out[u] for u in uids]


def _steer(head, refs, monkeypatch):
    """Replace the head's drafts in the first rounds so row j accepts
    _SCHEDULE[round][j]: the row's own next tokens, then one it cannot
    accept. The head still drafts, so its own cache moves as it would."""
    real = head.draft_block
    state = {"round": 0, "pos": [1] * len(refs)}

    def draft_block(last_bonus, hidden, cache, block_size, *a, **kw):
        drafts = real(last_bonus, hidden, cache, block_size, *a, **kw)
        r = state["round"]
        if r >= len(_SCHEDULE) or drafts.shape[0] != len(refs):
            return drafts
        state["round"] += 1
        n = drafts.shape[1]
        bonus = np.array(last_bonus).reshape(-1).tolist()
        rows = []
        for j, ref in enumerate(refs):
            pos = state["pos"][j]
            assert bonus[j] == ref[pos - 1], (r, j)
            a = min(_SCHEDULE[r][j], n)
            row = ref[pos:pos + a]
            if a < n:
                row = row + [(ref[pos + a] + 1) % 128] * (n - a)
            rows.append(row)
            state["pos"][j] = pos + a + 1
        return mx.array(rows, dtype=drafts.dtype)

    monkeypatch.setattr(head, "draft_block", draft_block)
    return state


def _count_route(monkeypatch):
    """Calls of the row ends route by query width, target and head."""
    seen = {"target": [], "head": []}
    for name, mod in (("target", qwen35_attn), ("head", qwen35_language)):
        inner = mod.row_ends_attention

        def spy(q, k, v, cache, scale, _inner=inner, _name=name):
            seen[_name].append(int(q.shape[2]))
            return _inner(q, k, v, cache, scale)

        monkeypatch.setattr(mod, "row_ends_attention", spy)
    return seen


@_metal
@_needs_ends
def test_uneven_rounds_keep_each_rows_tokens(serve, monkeypatch):  # noqa: F811
    install_owned_spec_engine()
    install_continuous_batch_admission()
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", str(len(_LENS)))
    spec._width_cap_memo = ("", None)
    model, head = _mtp_model()
    lm = model.language_model
    assert row_ends.row_ends_active(head, lm, False)
    rows = [_request(model, ids, []) for ids in _prompts()]
    alone = [_run(model, [row], head)[0] for row in rows]
    assert all(len(t) == _MAX for t in alone)

    state = _steer(head, alone, monkeypatch)
    seen = _count_route(monkeypatch)
    batch = _run(model, rows, head)

    assert state["round"] == len(_SCHEDULE)
    assert batch == alone
    # The verifies after the uneven rounds read each row up to its own end,
    # in the target and in the head's own layer.
    assert any(qL > 1 for qL in seen["target"])
    assert seen["head"]


# -- the gate ---------------------------------------------------------------


@_needs_ends
def test_only_the_dense_head_on_the_owned_dense_target_arms(monkeypatch):
    from gmlx.models.qwen35.owned import language_model_class
    from gmlx.spec.dflash_drafter import DFlash2Drafter

    model, head = _mtp_model()
    lm = model.language_model
    assert row_ends.row_ends_active(head, lm, False)
    assert not row_ends.row_ends_active(head, lm, True)        # mixed
    stock = qwen35_language.LanguageModel.__new__(
        qwen35_language.LanguageModel)
    assert not row_ends.row_ends_active(head, stock, False)
    moe = language_model_class("qwen3_5_moe")
    assert not row_ends.row_ends_active(head, moe.__new__(moe), False)
    assert not getattr(DFlash2Drafter, "row_ends_rollback", False)
    # A MoE head keeps the roll: the property reads the head's layer types.
    from gmlx.models.qwen35.moe_language import Qwen3_5MoeDecoderLayer
    from gmlx.spec.mtp_drafter import QwenMTPDrafter

    flag = QwenMTPDrafter.row_ends_rollback.fget
    assert flag(head)
    moe_layer = Qwen3_5MoeDecoderLayer.__new__(Qwen3_5MoeDecoderLayer)
    assert not flag(SimpleNamespace(layers=[moe_layer]))
    monkeypatch.setenv("GMLX_ROW_ENDS", "0")
    assert not row_ends.row_ends_active(head, lm, False)


class _ArmableDrafter(_StrictDrafter):
    row_ends_rollback = True


class _ArmableLM(_FakeLM):
    row_ends_rollback = True


def _uneven(pads=(0, 2, 1), n=12, accepted=(2, 0, 1), bs=3):
    """An armed batch cache whose rows end apart after a rollback, and each
    row's keys in order."""
    from gmlx.models.qwen35.language import rollback_attention_caches

    c = owned.BatchKVCache(list(pads))
    c._row_ends_armed = True
    k = mx.random.normal((len(pads), 2, n, 16), key=mx.random.key(1))
    c.update_and_fetch(k, k)
    vk = mx.random.normal((len(pads), 2, bs, 16), key=mx.random.key(2))
    c.update_and_fetch(vk, vk)
    rollback_attention_caches([c], list(accepted), bs)
    rows = [mx.concatenate([k[i:i + 1, :, p:], vk[i:i + 1, :, :a + 1]], axis=2)
            for i, (p, a) in enumerate(zip(pads, accepted))]
    assert c._row_ends is not None
    return c, rows


@_needs_ends
def test_a_gated_batch_never_arms(monkeypatch):
    """Past the cap the batch decodes plain and never arms; at the cap the
    same drafter and target arm before the first draft."""
    armed = []
    monkeypatch.setattr(row_ends, "arm", lambda caches: armed.append(caches))
    d = _ArmableDrafter(cap=2)
    _drive(d, B=3, max_tokens=3, lm=_ArmableLM())
    assert d.forward_calls == [] and armed == []
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(_ArmableDrafter(cap=3), B=3, max_tokens=3, lm=_ArmableLM())
    assert armed


@_needs_ends
def test_a_trip_right_justifies_the_rows():
    """A batch born speculating at the cap trips on an admission: the trip
    right-justifies its uneven rows and disarms the cache before the
    joiner lands and the plain step runs, each row holding its own keys."""
    c, rows = _uneven()
    d = _ArmableDrafter(cap=3)
    model = SimpleNamespace()
    lm = _ArmableLM()
    _queue_injection(model, lm)
    joiner = owned.KVCache()
    jk = mx.random.normal((1, 2, 5, 16), key=mx.random.key(3))
    joiner.update_and_fetch(jk, jk)
    model._generator_injections[0]["prompt_cache"] = [
        _FakeCache(width=1, offset=9), joiner]
    seen = []
    disarm = row_ends.disarm

    def spy(caches):
        before = getattr(caches[1], "_row_ends", None) if caches else None
        disarm(caches)
        if caches:
            x = caches[1]
            n = x._idx
            pads = x.left_padding.tolist()
            seen.append((before, x._row_ends, x._row_ends_armed, n, pads,
                         [np.asarray(x.keys[i, :, p:n]) for i, p in
                          enumerate(pads)]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(row_ends, "disarm", spy)
        _drive(d, B=3, max_tokens=4, model=model, lm=lm,
               prompt_cache=[_FakeCache(width=3), c])
    assert d.forward_calls == []
    assert len(seen) == 1
    before, after, armed, n, pads, held = seen[0]
    assert before is not None and after is None and not armed
    # The trip runs before the admission it counts, on the three rows.
    assert len(held) == len(rows)
    for i, k in enumerate(rows):
        np.testing.assert_array_equal(held[i], np.asarray(k[0]))
