"""The DeepSeek-V4 MTP undo log leaves other rotating caches stock.

ensure_rollback_attached wraps the rotating cache classes of every origin
for the rest of the process, and a served process can hold a DeepSeek-V4
model beside a Gemma-4 model. A cache that no DeepSeek-V4 verify armed
must behave as the stock class. Each check runs in a child that attached
the undo log and in a fresh child that never did, and compares the two.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

_UNDO_MODULE = "gmlx.models.deepseek_v4.cache"

CHILD = r'''
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)

# Serve loads the mlx-vlm runtime before a model builds, so the attach
# wraps both origins.
import mlx_vlm.models.cache  # noqa: F401

from gmlx.cache.compat import cache_types

if sys.argv[1] == "1":
    from gmlx.models.deepseek_v4.cache import ensure_rollback_attached

    ensure_rollback_attached()


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
        "wrapped": vars(type(cache))["trim"].__module__,
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


def state(cache, logits):
    return {
        "offsets": [value(c.offset) for c in cache],
        "idx": [getattr(c, "_idx", None) for c in cache],
        "logits": logits.tolist(),
    }


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
    out = {}

    # One row past the window: a verify block of three, both drafts
    # rejected, then the next token.
    cache = lm.make_cache()
    lm(mx.array(rows[:1]), cache=cache)
    lm(mx.array([[5, 9, 11]]), cache=cache)
    lm.rollback_speculative_cache(cache, None, 0, 3)
    out["one"] = state(cache, lm(mx.array([[9]]), cache=cache).logits[:, -1])

    # The same round on a batch of that row and a short one.
    singles = []
    for row in rows:
        c = lm.make_cache()
        lm(mx.array([row]), cache=c)
        singles.append(c)
    cache = [type(layer[0]).merge(list(layer)) for layer in zip(*singles)]
    lm(mx.array([[5, 9, 11], [6, 10, 12]]), cache=cache)
    lm.rollback_speculative_cache(cache, None, [0, 0], 3)
    out["two"] = state(
        cache, lm(mx.array([[9], [10]]), cache=cache).logits[:, -1])
    return out


print(json.dumps({"caches": caches(), "gemma": gemma()}))
'''


def _child(attach: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", CHILD, attach], capture_output=True,
        text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {"stock": _child("0"), "attached": _child("1")}


def test_unarmed_rotating_caches_trim_like_stock(runs):
    stock, attached = runs["stock"]["caches"], runs["attached"]["caches"]
    assert set(stock) == set(attached)
    assert len(stock) >= 2
    for name, want in stock.items():
        got = attached[name]
        assert want["wrapped"] != _UNDO_MODULE, name
        assert got["wrapped"] == _UNDO_MODULE, name
        # A full window: stock reports it untrimmable and trims anyway.
        assert not want["trimmable"] and want["trim"] == 2, name
        for key, v in want.items():
            if key != "wrapped":
                assert got[key] == v, (name, key)


@pytest.mark.parametrize("rows", ["one", "two"])
def test_gemma4_rollback_after_a_deepseek_v4_attach(runs, rows):
    want = runs["stock"]["gemma"][rows]
    got = runs["attached"]["gemma"][rows]
    assert got["offsets"] == want["offsets"]
    assert got["idx"] == want["idx"]
    d = np.abs(np.array(got["logits"]) - np.array(want["logits"])).max()
    assert d <= 1e-6, d
