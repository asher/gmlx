"""A trim of a rotating cache in temporal order drops the trimmed columns.

Each key carries its position and a tag: 0 for the prompt, 99 for a
rejected draft, 1 for the token written after the rollback, 5 for a row
that joins later. After a verify block past a full window and a trim of
its two drafts, a one-token step must attend what a single-row cache fed
only the kept history attends. The composition checks run in children
that install the guard and the DeepSeek-V4 undo log in either order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

W = 8


def _keys(cols, tags, rows=1):
    k = mx.stack([mx.array(cols, dtype=mx.float32) + 1,
                  mx.array(tags, dtype=mx.float32)], axis=-1)
    return mx.broadcast_to(k[None, None], (rows, 1, len(cols), 2))


def _seen(cache, k):
    """The (position, tag) pairs each row's query attends in a one-token
    update with keys ``k``."""
    batch = hasattr(cache, "rotated")
    mask = cache.make_mask(1) if batch else None
    kk, _ = cache.update_and_fetch(k, k)
    out = []
    for b in range(k.shape[0]):
        if mask is None:
            vis = np.ones(kk.shape[2], dtype=bool)
        else:
            full = mx.broadcast_to(mask, (k.shape[0], 1, 1, kk.shape[2]))
            vis = np.array(full)[b, 0, 0]
        cols = np.array(kk[b, 0, :, 0]).astype(int) - 1
        tags = np.array(kk[b, 0, :, 1]).astype(int)
        out.append(sorted((int(c), int(t))
                          for c, t, v in zip(cols, tags, vis) if v))
    return out


def _truth(single, cols, tags, col, tag):
    """What a single-row cache fed ``cols`` attends at ``col``."""
    c = single(max_size=W)
    k = _keys(cols, tags)
    c.update_and_fetch(k, k)
    return _seen(c, _keys([col], [tag]))[0]


def _rolled_back(cache, rows, rejected=2):
    """Prompt 0..11, a one-token step at 12, a verify block 13..15 whose
    last ``rejected`` columns are drafts, then the rollback trim."""
    for cols, tags in ((list(range(12)), [0] * 12), ([12], [0]),
                       ([13, 14, 15], [0] * (3 - rejected) + [99] * rejected)):
        k = _keys(cols, tags, rows)
        cache.update_and_fetch(k, k)
    assert cache.trim(rejected) == rejected


def _classes():
    from gmlx.cache.compat import cache_types

    return list(zip(cache_types("RotatingKVCache"),
                    cache_types("BatchRotatingKVCache")))


def _one_token_after_trim(single, batch, kind):
    """(seen, truth) for every row of a one-token step after the trim."""
    pads = {"single": None, "batch-1": [0], "batch-2": [0, 9]}[kind]
    cache = single(max_size=W) if pads is None else batch(W, pads)
    rows = 1 if pads is None else len(pads)
    _rolled_back(cache, rows)
    seen = _seen(cache, _keys([14], [1], rows))
    truth = [_truth(single, list(range(p, 14)), [0] * (14 - p), 14, 1)
             for p in (pads or [0])]
    return seen, truth


def _extend_then_one_token(single, batch):
    cache = batch(W, [0, 9])
    _rolled_back(cache, 2)
    joining = single(max_size=W)
    k = _keys(list(range(5)), [5] * 5)
    joining.update_and_fetch(k, k)
    cache.extend(batch.merge([joining]))
    k = mx.concatenate([_keys([14], [1], 2), _keys([5], [1])], axis=0)
    seen = _seen(cache, k)
    truth = [_truth(single, list(range(p, 14)), [0] * (14 - p), 14, 1)
             for p in (0, 9)]
    truth.append(_truth(single, list(range(5)), [5] * 5, 5, 1))
    return seen, truth


def _armed_undo(single, batch, kind, rejected):
    """The DeepSeek-V4 undo log's path: an armed verify, then its trim."""
    from gmlx.models.deepseek_v4.cache import set_undo_armed

    pads = None if kind == "single" else [0, 9]
    cache = single(max_size=W) if pads is None else batch(W, pads)
    rows = 1 if pads is None else 2
    for cols, tags in ((list(range(12)), [0] * 12), ([12], [0])):
        k = _keys(cols, tags, rows)
        cache.update_and_fetch(k, k)
    set_undo_armed(True)
    try:
        k = _keys([13, 14, 15], [0] * (3 - rejected) + [99] * rejected, rows)
        cache.update_and_fetch(k, k)
    finally:
        set_undo_armed(False)
    assert cache.trim(rejected) == rejected
    col = 16 - rejected
    seen = _seen(cache, _keys([col], [1], rows))
    truth = [_truth(single, list(range(p, col)), [0] * (col - p), col, 1)
             for p in (pads or [0])]
    return seen, truth


@pytest.fixture
def guarded():
    import mlx_vlm.models.cache  # noqa: F401

    from gmlx.upstream.batch_rotating_mask_fix import (
        install_batch_rotating_mask_fix,
    )
    from gmlx.upstream.rotating_trim_fix import install_rotating_trim_fix

    # The loader installs both with every model.
    install_batch_rotating_mask_fix()
    install_rotating_trim_fix()
    return _classes()


@pytest.mark.parametrize("kind", ["single", "batch-1", "batch-2"])
def test_a_one_token_step_after_a_trim_matches_a_single_row(guarded, kind):
    assert len(guarded) == 2
    for single, batch in guarded:
        seen, truth = _one_token_after_trim(single, batch, kind)
        assert seen == truth, (single.__module__, kind)


def test_an_admission_then_a_one_token_step_matches_a_single_row(guarded):
    for single, batch in guarded:
        seen, truth = _extend_then_one_token(single, batch)
        assert seen == truth, single.__module__


def test_a_trim_below_the_window_keeps_the_growing_buffer(guarded):
    for _single, batch in guarded:
        cache = batch(W, [0])
        k = _keys(list(range(4)), [0] * 4)
        cache.update_and_fetch(k, k)
        k = _keys([4, 5, 6], [0, 99, 99])
        cache.update_and_fetch(k, k)
        assert cache.trim(2) == 2
        for col in (5, 6, 7, 8):
            seen = _seen(cache, _keys([col], [1]))[0]
            want = [(c, 0) for c in range(max(0, col - W + 1), 5)]
            want += [(c, 1) for c in range(5, col + 1)]
            assert seen == want, (batch.__module__, col)


def test_a_trim_deeper_than_the_update_keeps_the_stock_buffer(guarded):
    # A cut to _idx below the window would make _update_in_place grow the
    # buffer from an offset past it, which raises.
    for single, batch in guarded:
        for cache in (single(max_size=W), batch(W, [0])):
            for cols in (list(range(12)), [12], [13, 14, 15]):
                k = _keys(cols, [0] * len(cols))
                cache.update_and_fetch(k, k)
            assert cache.trim(3) == 3
            assert cache.keys.shape[2] == W + 2
            assert len(_seen(cache, _keys([13], [1]))[0]) == W


def test_install_is_idempotent_and_has_a_kill_switch(guarded, monkeypatch):
    from gmlx.upstream.rotating_trim_fix import install_rotating_trim_fix

    before = [(s.__dict__["trim"], b.__dict__["trim"]) for s, b in guarded]
    assert install_rotating_trim_fix()
    assert [(s.__dict__["trim"], b.__dict__["trim"])
            for s, b in guarded] == before
    monkeypatch.setenv("GMLX_ROTATING_TRIM_FIX", "0")
    assert install_rotating_trim_fix() is False


def _child(order: str) -> dict:
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), order],
        capture_output=True, text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("order", ["guard-first", "undo-first"])
def test_the_guard_composes_with_the_deepseek_v4_undo_log(order):
    out = _child(order)
    assert out["wrapped"], out
    assert out["cases"], out
    for name, (seen, truth) in out["cases"].items():
        assert seen == truth, (order, name)


def _main(order: str) -> None:
    mx.set_default_device(mx.cpu)
    import mlx_vlm.models.cache  # noqa: F401

    from gmlx.models.deepseek_v4.cache import ensure_rollback_attached
    from gmlx.upstream.batch_rotating_mask_fix import (
        install_batch_rotating_mask_fix,
    )
    from gmlx.upstream.rotating_trim_fix import install_rotating_trim_fix

    install_batch_rotating_mask_fix()
    if order == "guard-first":
        install_rotating_trim_fix()
        ensure_rollback_attached()
    else:
        ensure_rollback_attached()
        install_rotating_trim_fix()
    classes = _classes()
    wrapped = all(
        cls.__dict__.get("_gmlx_trim_drops_tail")
        and cls.__dict__.get("_gmlx_mtp_undo_attached")
        for pair in classes for cls in pair)
    cases = {}
    for single, batch in classes:
        tag = single.__module__
        for kind in ("single", "batch-1", "batch-2"):
            cases[f"{tag} unarmed {kind}"] = _one_token_after_trim(
                single, batch, kind)
        cases[f"{tag} unarmed extend"] = _extend_then_one_token(single, batch)
        for kind in ("single", "batch"):
            for rejected in (1, 2):
                cases[f"{tag} armed {kind} trim {rejected}"] = _armed_undo(
                    single, batch, kind, rejected)
    print(json.dumps({"wrapped": bool(wrapped), "cases": cases}))


if __name__ == "__main__":
    _main(sys.argv[1])
