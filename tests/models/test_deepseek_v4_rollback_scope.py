"""The DeepSeek-V4 MTP undo log leaves other rotating caches as they are.

The owned rotating cache classes carry the undo log for the whole
process, and a served process can hold a DeepSeek-V4 model beside a
Gemma-4 model. A cache that no DeepSeek-V4 verify armed must behave as the
class without the undo log, and a verify armed on one thread must not arm
a cache updated on another. Each check runs in a fresh child with the undo
log and in a child where it never stashes. The children must agree, and
each Gemma-4 rollback must match a single-row forward over the kept
tokens, whether a one-token step or a verify-width block follows it. A
control child runs the stock trim bodies, where the one-token step must
miss that forward, so the check can still tell a broken rollback.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

CHILD = r'''
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx_vlm.models.cache  # noqa: F401

import gmlx.cache.kv_cache as owned
from gmlx.cache import bind

CLASSES = (owned.RotatingKVCache, owned.BatchRotatingKVCache)
mode = sys.argv[1]
if mode == "no-undo":
    owned._stash_undo = lambda *args: None
elif mode == "stock-trim":
    for cls in CLASSES:
        stock = next(old for old, new in bind.replaced()
                     if new is cls and old.__module__ == "mlx_vlm.models.cache")
        cls.trim = vars(stock)["trim"]


def value(v):
    return v.tolist() if isinstance(v, mx.array) else v


def block(B, S, step):
    return mx.arange(B * S * 4, dtype=mx.float32).reshape(B, 1, S, 4) + 100 * step


def fill(cache, B):
    for step in range(12):
        k = block(B, 1, step)
        cache.update_and_fetch(k, k)
    k = block(B, 3, 12)
    cache.update_and_fetch(k, k)
    trimmable = cache.is_trimmable()
    got = cache.trim(2)
    return {
        "stash": "_mtp_undo" in vars(cache),
        "trimmable": trimmable, "trim": got,
        "offset": value(cache.offset), "idx": cache._idx,
        "_offset": value(getattr(cache, "_offset", None)),
        "left_padding": value(getattr(cache, "left_padding", None)),
        "keys": cache.keys.tolist(),
    }


def caches():
    single, batch = CLASSES
    return {single.__name__: fill(single(max_size=8), 1),
            batch.__name__: fill(batch(max_size=8, left_padding=[2, 0]), 2)}


def gemma():
    from mlx_vlm.models.gemma4.config import TextConfig

    import gmlx.models.gemma4.sync as gemma4_sync
    from gmlx.models.gemma4.owned import OwnedGemma4LanguageModel

    gemma4_sync.install_gemma4_nosync()
    cfg = TextConfig(
        model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
        intermediate_size=128, num_attention_heads=4, head_dim=16,
        global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
        vocab_size_per_layer_input=128, num_key_value_heads=2,
        num_kv_shared_layers=2, hidden_size_per_layer_input=0,
        sliding_window=16, sliding_window_pattern=3,
        tie_word_embeddings=True)
    mx.random.seed(13)
    lm = OwnedGemma4LanguageModel(cfg)
    mx.eval(lm.parameters())
    rows = [[(7 * i + 3) % 127 + 1 for i in range(30)],
            [(5 * i + 2) % 127 + 1 for i in range(6)]]
    # Each row's verify block rejects both drafts; the first column stays.
    verify, keep = [[5, 9, 11], [6, 10, 12]], [[5, 9], [6, 10]]

    def truth(tokens):
        out = lm(mx.array([tokens]), cache=lm.make_cache())
        return out.logits[0, -1].tolist()

    def rolled_back(n):
        singles = []
        for row in rows[:n]:
            c = lm.make_cache()
            lm(mx.array([row]), cache=c)
            singles.append(c)
        if n == 1:
            cache = singles[0]
        else:
            cache = [type(layer[0]).merge(list(layer))
                     for layer in zip(*singles)]
        lm(mx.array(verify[:n]), cache=cache)
        lm.rollback_speculative_cache(cache, None, 0 if n == 1 else [0] * n, 3)
        return cache

    out = {}
    for name, n in (("one", 1), ("two", 2)):
        want = [truth(rows[r] + keep[r]) for r in range(n)]
        # The next round is a one-token step, as after the width gate, or a
        # verify block whose first column is scored.
        for follow, ids in (("step", [[k[1]] for k in keep[:n]]),
                            ("block", [[k[1], 4] for k in keep[:n]])):
            cache = rolled_back(n)
            logits = lm(mx.array(ids), cache=cache).logits[:, 0]
            out[f"{name} {follow}"] = {
                "offsets": [value(c.offset) for c in cache],
                "idx": [getattr(c, "_idx", None) for c in cache],
                "logits": logits.tolist(), "truth": want,
            }
    return out


print(json.dumps({"caches": caches(), "gemma": gemma()}))
'''

_MODES = ("owned", "no-undo")
# The control: the stock trim bodies.
_CONTROL = "stock-trim"


def _child(mode: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", CHILD, mode], capture_output=True,
        text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {mode: _child(mode) for mode in _MODES + (_CONTROL,)}


def test_unarmed_rotating_caches_trim_as_without_the_undo_log(runs):
    plain, owned = runs["no-undo"]["caches"], runs["owned"]["caches"]
    assert set(plain) == set(owned) == {"RotatingKVCache",
                                        "BatchRotatingKVCache"}
    for name, want in plain.items():
        got = owned[name]
        assert not want["stash"] and not got["stash"], name
        # A full window reports itself untrimmable and trims anyway.
        assert not want["trimmable"] and want["trim"] == 2, name
        assert got == want, name


def test_a_verify_armed_on_another_thread_arms_nothing_here():
    import threading

    import mlx.core as mx

    from gmlx.cache.kv_cache import RotatingKVCache, set_undo_armed

    armed, done = threading.Event(), threading.Event()

    def verify():
        set_undo_armed(True)
        armed.set()
        done.wait(10)
        set_undo_armed(False)

    t = threading.Thread(target=verify)
    t.start()
    try:
        assert armed.wait(10)
        cache = RotatingKVCache(max_size=4)
        for _ in range(6):
            cache.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
        cache.update_and_fetch(mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))
        assert cache._mtp_undo is None and not cache.is_trimmable()
    finally:
        done.set()
        t.join()


_CASES = [f"{n} {f}" for n in ("one", "two") for f in ("step", "block")]


def _misses(got) -> list[float]:
    """Each row's largest logit miss against the fresh forward, as a
    fraction of the logit scale."""
    out = []
    for row, want in zip(got["logits"], got["truth"]):
        want = np.array(want)
        d = float(np.abs(np.array(row) - want).max())
        out.append(d / float(np.abs(want).max()))
    return out


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("mode", _MODES)
def test_gemma4_rollback_matches_a_single_row(runs, mode, case):
    misses = _misses(runs[mode]["gemma"][case])
    # A fresh forward agrees to about 1e-6 of the logit scale; rejected
    # drafts left attended move the logits by about the scale.
    assert all(m <= 1e-4 for m in misses), (mode, case, misses)


@pytest.mark.parametrize("case", [c for c in _CASES if c.endswith("step")])
def test_the_check_fails_with_the_stock_trim(runs, case):
    misses = _misses(runs[_CONTROL]["gemma"][case])
    assert all(m > 0.1 for m in misses), (case, misses)


@pytest.mark.parametrize("case", _CASES)
def test_gemma4_rollback_is_the_same_without_the_undo_log(runs, case):
    want = runs["no-undo"]["gemma"][case]
    got = runs["owned"]["gemma"][case]
    assert got["offsets"] == want["offsets"]
    assert got["idx"] == want["idx"]
    d = np.abs(np.array(got["logits"]) - np.array(want["logits"])).max()
    assert d <= 1e-6, d
