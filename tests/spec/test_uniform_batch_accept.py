"""Uniform batched accepts: a target whose rollback cannot trim ragged rows
gets every row clamped to the smallest accept count, with the target's own
token at that position as each row's bonus.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.spec.speculative import _coupled_walk_batch, _uniform_batch_accept

# Row 0 accepts 2 drafts, row 1 none, row 2 all 3.
DRAFTS = [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
TARGET = [[5, 6, 20, 21], [30, 31, 32, 33], [11, 12, 13, 40]]


def _walk(uniform, budgets=(99, 99, 99), drafts=DRAFTS, target=TARGET):
    verify = SimpleNamespace(target_tokens=mx.array(target), hidden=None)
    return _coupled_walk_batch(None, verify, mx.array(drafts), None,
                               list(budgets), uniform=uniform)


def test_ragged_accepts_without_uniform():
    acc, new = _walk(False)
    assert acc == [2, 0, 3]
    assert new == [[5, 6, 20], [30], [11, 12, 13, 40]]


def test_uniform_clamps_to_smallest_accept():
    acc, new = _walk(True)
    assert acc == [0, 0, 0]
    assert new == [[5], [30], [11]]


def test_uniform_respects_row_budget():
    drafts = [[5, 6, 7], [8, 9, 10]]
    target = [[5, 6, 20, 21], [8, 31, 32, 33]]
    acc, new = _walk(True, budgets=(1, 99), drafts=drafts, target=target)
    assert acc == [1, 1]
    assert new == [[5], [8, 31]]


def test_uniform_clamps_sampled_verify():
    """With a sampler the walk compares per-position target samples, and
    the clamp cuts it at the same place."""
    lm = SimpleNamespace(speculative_logits_from_hidden=lambda h: h)
    onehot = mx.arange(64)[None, None, :] == mx.array(TARGET)[:, :, None]
    verify = SimpleNamespace(target_tokens=None,
                             hidden=onehot.astype(mx.float32) * 10.0)
    acc, new = _coupled_walk_batch(lm, verify, mx.array(DRAFTS),
                                   lambda flat: mx.argmax(flat, axis=-1),
                                   [99, 99, 99], uniform=True)
    assert acc == [0, 0, 0]
    assert new == [[5], [30], [11]]


def test_uniform_leaves_equal_accepts_alone():
    drafts = [[5, 6, 7], [8, 9, 10]]
    target = [[5, 20, 21, 22], [8, 30, 31, 32]]
    assert _walk(True, drafts=drafts, target=target) == \
        _walk(False, drafts=drafts, target=target)


@pytest.mark.parametrize("where", ["drafter", "lm"])
def test_flag_on_drafter_or_target_clamps(where):
    flagged = SimpleNamespace(requires_uniform_batch_acceptance=True)
    plain = SimpleNamespace()
    drafter, lm = (flagged, plain) if where == "drafter" else (plain, flagged)
    assert _uniform_batch_accept(drafter, lm)


def test_gemma4_target_clamps_and_others_do_not():
    gemma4 = pytest.importorskip("mlx_vlm.models.gemma4.language")
    lm = gemma4.LanguageModel.__new__(gemma4.LanguageModel)
    assert _uniform_batch_accept(SimpleNamespace(), lm)
    assert not _uniform_batch_accept(SimpleNamespace(), SimpleNamespace())


def test_owned_gemma4_target_clamps_through_the_rollback_guard():
    """The class gmlx loads inherits the gemma4 hook, and the load path
    rebinds the hook on the instance."""
    from gmlx.gen.generation import harden_mtp_rollback
    from gmlx.models.gemma4.owned import OwnedGemma4LanguageModel

    lm = OwnedGemma4LanguageModel.__new__(OwnedGemma4LanguageModel)
    harden_mtp_rollback(lm)
    assert getattr(lm.rollback_speculative_cache, "_gmlx_kvarn_guard", False)
    assert _uniform_batch_accept(SimpleNamespace(), lm)
