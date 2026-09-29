"""Probes of a Gemma 4 text module, owned or stock, on tiny random models.

The owned Gemma 4 tests call each probe on ``gmlx.models.gemma4.language``
in process. Run as a script, this file calls every probe on
``mlx_vlm.models.gemma4.language`` in a child that imports gmlx, whose
cache binding gives stock gemma4 the owned caches and SDPA, and runs no
loader install, so the stock classes run their mlx-vlm 0.7.3 bodies. It
also compares owned and stock where the owned copy keeps the stock
answer. It prints one JSON object: probe -> result.
"""

from __future__ import annotations

import json
import sys

import mlx.core as mx
from gmlx._vendor.mlx_vlm.generate import ar
from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig

import gmlx  # noqa: F401  (the cache binding)

BASE = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
    vocab_size_per_layer_input=128, num_key_value_heads=2,
    num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    sliding_window=32, sliding_window_pattern=3, tie_word_embeddings=True,
)
PROMPT = [3, 17, 42, 99, 7, 63, 5, 28]


def tiny(lang, **over):
    mx.random.seed(13)
    lm = lang.LanguageModel(TextConfig(**{**BASE, **over}))
    mx.eval(lm.parameters())
    return lm


def _ids(toks, rows=1):
    return mx.array([list(toks)] * rows)


def _full(tm, cache):
    return list(cache) + [None] * (len(tm.layers) - len(cache))


class _Items:
    """Counts ``mx.array.item`` calls, each a host sync."""

    def __enter__(self):
        self.n = 0
        self._orig = mx.array.item

        def item(a):
            self.n += 1
            return self._orig(a)

        mx.array.item = item
        return self

    def __exit__(self, *exc):
        mx.array.item = self._orig


def _warm_caches(lm):
    """(int-offset caches, batch caches), each after a prompt."""
    single = lm.make_cache()
    mx.eval(lm(_ids(PROMPT), cache=single).logits)
    batch = ar._make_cache(lm, [0, 2])
    mx.eval(lm(_ids(PROMPT, 2), cache=batch).logits)
    return single, batch


def mask_syncs(lang) -> dict:
    """Host syncs one ``_make_masks`` call takes on a three-token chunk,
    by offset kind."""
    lm = tiny(lang)
    tm = lm.model
    out = {}
    for kind, cache in zip(("int", "array"), _warm_caches(lm)):
        rows = 2 if kind == "array" else 1
        with _Items() as items:
            tm._make_masks(mx.zeros((rows, 3, 64)), _full(tm, cache))
        out[kind] = items.n
    return out


class _Cache:
    left_padding = None

    def __init__(self, offset):
        self.offset = offset

    def update_and_fetch(self, k, v):
        return k, v


def rope_offsets(lang) -> dict:
    """What the attention rope receives: for an int cache offset, the
    kind and value of each offset; for an array offset, whether each is a
    copy with the same values."""
    lm = tiny(lang)
    attn = lm.model.layers[0].self_attn
    seen = []
    rope = attn.rope

    def spy(x, offset=0):
        seen.append(offset)
        return rope(x, offset=0)

    attn.rope = spy
    attn(mx.zeros((1, 1, 64)), cache=_Cache(9))
    ints = [[type(o).__name__, o if isinstance(o, int) else o.item()]
            for o in seen]
    seen.clear()
    live = mx.array([5, 6, 7])
    attn(mx.zeros((3, 1, 64)), cache=_Cache(live))
    copies = [isinstance(o, mx.array) and o is not live
              and bool(mx.array_equal(o, live).item()) for o in seen]
    return {"int": ints, "array_copied": copies}


def _same_masks(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if type(x) is not type(y):
            return False
        if isinstance(x, mx.array):
            if not bool(mx.array_equal(x, y).item()):
                return False
        elif x != y:
            return False
    return True


def mask_decisions(stock, owned) -> dict:
    """Owned and stock ``_make_masks`` on int offsets: a prefill from
    empty, a verify block and a decode step. True where they agree."""
    lm_s, lm_o = tiny(stock), tiny(owned)
    out = {}
    fresh_s, fresh_o = lm_s.make_cache(), lm_o.make_cache()
    out["prefill"] = _same_masks(
        lm_s.model._make_masks(mx.zeros((1, 8, 64)), _full(lm_s.model, fresh_s)),
        lm_o.model._make_masks(mx.zeros((1, 8, 64)), _full(lm_o.model, fresh_o)))
    warm_s, _ = _warm_caches(lm_s)
    warm_o, _ = _warm_caches(lm_o)
    for name, width in (("verify", 3), ("decode", 1)):
        h = mx.zeros((1, width, 64))
        out[name] = _same_masks(
            lm_s.model._make_masks(h, _full(lm_s.model, warm_s)),
            lm_o.model._make_masks(h, _full(lm_o.model, warm_o)))
    return out


def main():
    from gmlx._vendor.mlx_vlm.models.gemma4 import language as stock

    from gmlx.models.gemma4 import language as owned

    mx.set_default_device(mx.cpu if "--cpu" in sys.argv else mx.gpu)
    result = {
        "mask_syncs": mask_syncs(stock),
        "rope_offsets": rope_offsets(stock),
        "mask_decisions": mask_decisions(stock, owned),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
