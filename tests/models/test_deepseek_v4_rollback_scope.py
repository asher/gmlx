"""The DeepSeek-V4 MTP undo log leaves other rotating caches stock.

ensure_rollback_attached wraps the rotating cache classes of every origin
for the rest of the process, and a served process can hold a DeepSeek-V4
model beside a Gemma-4 model. A cache that no DeepSeek-V4 verify armed
must behave as the stock class. Each check runs in a fresh child that never
attached the undo log and in children that attached it before and after the
stock trim fix, as a DeepSeek-V4 load and a Gemma-4 load install them in
either order. The children must agree, and each Gemma-4 rollback must match
a single-row forward over the kept tokens, whether a one-token step or a
verify-width block follows it.
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

# Serve loads the mlx-vlm runtime before a model builds, so the attach
# wraps both origins.
import mlx_vlm.models.cache  # noqa: F401

from gmlx.cache.compat import cache_types
from gmlx.upstream.rotating_trim_fix import install_rotating_trim_fix


def attach():
    from gmlx.models.deepseek_v4.cache import ensure_rollback_attached

    ensure_rollback_attached()


# The loader installs the trim fix with every model; a DeepSeek-V4 model
# attaches the undo log as it builds, before that install.
order = {"stock": [install_rotating_trim_fix],
         "undo-first": [attach, install_rotating_trim_fix],
         "guard-first": [install_rotating_trim_fix, attach]}[sys.argv[1]]
for step in order:
    step()


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
        "undo": bool(vars(type(cache)).get("_gmlx_mtp_undo_attached")),
        "trimmable": trimmable, "trim": got,
        "offset": value(cache.offset), "idx": cache._idx,
        "_offset": value(getattr(cache, "_offset", None)),
        "left_padding": value(getattr(cache, "left_padding", None)),
        "keys": cache.keys.tolist(),
    }


def caches():
    out = {}
    for cls in cache_types("RotatingKVCache"):
        out[f"{cls.__module__}.{cls.__name__}"] = fill(cls(max_size=8), 1)
    for cls in cache_types("BatchRotatingKVCache"):
        out[f"{cls.__module__}.{cls.__name__}"] = fill(
            cls(max_size=8, left_padding=[2, 0]), 2)
    return out


def gemma():
    from mlx_vlm.models.gemma4.config import TextConfig

    import gmlx.models.gemma4.sync as gemma4_sync
    from gmlx.models.gemma4.owned import OwnedGemma4LanguageModel
    from gmlx.upstream.batch_rotating_mask_fix import (
        install_batch_rotating_mask_fix,
    )

    gemma4_sync.install_gemma4_nosync()
    install_batch_rotating_mask_fix()
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

_ORDERS = ("undo-first", "guard-first")


def _child(order: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", CHILD, order], capture_output=True,
        text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {order: _child(order) for order in ("stock",) + _ORDERS}


@pytest.mark.parametrize("order", _ORDERS)
def test_unarmed_rotating_caches_trim_like_stock(runs, order):
    stock, attached = runs["stock"]["caches"], runs[order]["caches"]
    assert set(stock) == set(attached)
    assert len(stock) >= 4
    for name, want in stock.items():
        got = attached[name]
        assert not want["undo"] and got["undo"], name
        # A full window: stock reports it untrimmable and trims anyway.
        assert not want["trimmable"] and want["trim"] == 2, name
        for key, v in want.items():
            if key != "undo":
                assert got[key] == v, (name, key)


_CASES = [f"{n} {f}" for n in ("one", "two") for f in ("step", "block")]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("order", ("stock",) + _ORDERS)
def test_gemma4_rollback_matches_a_single_row(runs, order, case):
    got = runs[order]["gemma"][case]
    for row, want in zip(got["logits"], got["truth"]):
        want = np.array(want)
        d = float(np.abs(np.array(row) - want).max())
        # A fresh forward agrees to about 1e-6 of the logit scale; rejected
        # drafts left attended move the logits by about the scale.
        assert d <= 1e-4 * float(np.abs(want).max()), (order, case, d)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("order", _ORDERS)
def test_gemma4_rollback_after_a_deepseek_v4_attach(runs, order, case):
    want = runs["stock"]["gemma"][case]
    got = runs[order]["gemma"][case]
    assert got["offsets"] == want["offsets"]
    assert got["idx"] == want["idx"]
    d = np.abs(np.array(got["logits"]) - np.array(want["logits"])).max()
    assert d <= 1e-6, d
