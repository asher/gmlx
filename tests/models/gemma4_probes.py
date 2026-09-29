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


IMAGE_WINDOW, IMAGE_BLOCKS, IMAGE_N = 8, [(6, 20), (24, 34)], 40


def image_types(blocks=IMAGE_BLOCKS, n=IMAGE_N) -> mx.array:
    row = [0] * n
    for a, b in blocks:
        row[a:b] = [1] * (b - a)
    return mx.array([row], dtype=mx.int32)


def image_mask_spread(lang) -> dict:
    """Where one pass over a prompt with two image blocks, under
    bidirectional vision and a window narrower than each block, lets a
    query see a key: whether the full-attention mask is causal, whether a
    sliding mask reaches a key past the window, and whether it lets a
    block token see a later token of its block."""
    lm = tiny(lang, use_bidirectional_attention="vision",
              sliding_window=IMAGE_WINDOW)
    tm = lm.model
    n = IMAGE_N
    masks = tm._make_masks(mx.zeros((1, n, 64)), [None] * len(tm.layers),
                           image_types())
    q = mx.arange(n)[:, None]
    k = mx.arange(n)[None, :]
    causal = k <= q
    out = {}
    for layer, mask in zip(tm.layers, masks):
        if layer.layer_type in out:
            continue
        if isinstance(mask, str):
            mask = causal
        mask = mx.broadcast_to(mask, (1, 1, n, n))[0, 0]
        if layer.layer_type == "full_attention":
            out["full_attention"] = {
                "causal": bool(mx.array_equal(mask, causal).item())}
        else:
            past = mask & (k <= q - IMAGE_WINDOW)
            out["sliding_attention"] = {
                "past_window": bool(mx.any(past).item()),
                "block_ahead": bool(mask[6, 12].item()),
            }
    return out


POLICY_PROMPTS = {"image": (True, False), "image_audio": (True, True),
                  "audio": (False, True), "text": (False, False)}


def policy_prompt(image, audio, image_id=1, audio_id=2):
    """(ids, token types) of a 16-token prompt with an image at 4-9 and
    audio at 12-14."""
    ids, types = [], []
    for i in range(16):
        if image and 4 <= i < 10:
            ids.append(image_id)
            types.append(1)
        elif audio and 12 <= i < 15:
            ids.append(audio_id)
            types.append(3)
        else:
            ids.append(20 + i)
            types.append(0)
    return mx.array([ids]), mx.array([types])


def lm_policy(lang) -> dict:
    """The language model's chunking answer under bidirectional vision,
    by prompt, without and with an MTP drafter, and the host syncs the
    two calls take."""
    lm = tiny(lang, use_bidirectional_attention="vision")
    draft = object()
    out = {}
    for name, (image, audio) in POLICY_PROMPTS.items():
        _ids, types = policy_prompt(image, audio)
        kw = {"mm_token_type_ids": types}
        with _Items() as items:
            plain = lm.chunked_prefill_policy(prefill_kwargs=kw)
            mtp = lm.chunked_prefill_policy(
                draft_model=draft, draft_kind="mtp",
                prefill_kwargs={**kw, "return_hidden": True,
                                "return_shared_kv": True})
        out[name] = {"plain": plain, "mtp": mtp, "syncs": items.n}
    return out


def unified(uni):
    """A tiny gemma4_unified container of the module ``uni``."""
    from gmlx._vendor.mlx_vlm.models.gemma4_unified.config import ModelConfig

    cfg = ModelConfig.from_dict({
        "model_type": "gemma4_unified",
        "text_config": {**BASE, "model_type": "gemma4_unified_text"},
        "vision_config": {}})
    return uni.Model(cfg)


def unified_policy(uni) -> dict:
    """The gemma4_unified container's chunking answer, by prompt, from the
    token types and from the ids alone, and the host syncs the two calls
    take."""
    model = unified(uni)
    c = model.config
    out = {}
    for name, (image, audio) in POLICY_PROMPTS.items():
        ids, types = policy_prompt(image, audio, c.image_token_id,
                                   c.audio_token_id)
        with _Items() as items:
            by_types = model.chunked_prefill_policy(
                input_ids=ids, prefill_kwargs={"mm_token_type_ids": types})
            by_ids = model.chunked_prefill_policy(input_ids=ids,
                                                  prefill_kwargs={})
        out[name] = {"by_types": by_types, "by_ids": by_ids,
                     "syncs": items.n}
    return out


SOFTCAP = 30.0


def softcap(fn) -> dict:
    """The dtype of ``fn(SOFTCAP, x)`` on bfloat16 logits, and its largest
    error against the float32 softcap."""
    mx.random.seed(3)
    x = (mx.random.normal((2, 4096)) * 12.0).astype(mx.bfloat16)
    out = fn(SOFTCAP, x)
    ref = mx.tanh(x.astype(mx.float32) / SOFTCAP) * SOFTCAP
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    return {"dtype": str(out.dtype), "err": err}


def model_logits_dtype(lang) -> str:
    """The logits dtype of a bfloat16 model with a final softcap."""
    lm = tiny(lang, final_logit_softcapping=SOFTCAP)
    lm.set_dtype(mx.bfloat16)
    return str(lm(_ids(PROMPT)).logits.dtype)


PLE = dict(num_hidden_layers=4, num_kv_shared_layers=2,
           hidden_size_per_layer_input=8, sliding_window_pattern=2)
PLE_ROWS = [list(range(1, 21)), list(range(30, 44))]


def ple_batch_inputs(lm):
    """Left-padded ids for PLE_ROWS, their embeddings, per-layer inputs
    and a batch cache after a four-column chunk."""
    n = max(len(r) for r in PLE_ROWS)
    pads = [n - len(r) for r in PLE_ROWS]
    ids = mx.array([[0] * p + r for p, r in zip(pads, PLE_ROWS)])
    embeds = lm.model.embed_tokens(ids) * lm.model.embed_scale
    pli = lm.model.get_per_layer_inputs(ids)
    cache = ar._make_cache(lm, pads)
    mx.eval(lm(None, inputs_embeds=embeds[:, :4], cache=cache,
               per_layer_inputs=pli[:, :4]).logits)
    return embeds, pli, cache


def ple_batch_slice(lang) -> str:
    """What the text model does with whole-prompt per-layer inputs and a
    later chunk on a batch cache: "ok", or the exception's type."""
    lm = tiny(lang, **PLE)
    embeds, pli, cache = ple_batch_inputs(lm)
    try:
        mx.eval(lm(None, inputs_embeds=embeds[:, 4:8], cache=cache,
                   per_layer_inputs=pli).logits)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return "ok"


def main():
    from gmlx._vendor.mlx_lm.models import gemma4_text as stock_lm
    from gmlx._vendor.mlx_vlm.models.gemma4 import language as stock
    from gmlx._vendor.mlx_vlm.models.gemma4_unified import gemma4_unified as stock_uni

    from gmlx.models.gemma4 import language as owned

    mx.set_default_device(mx.cpu if "--cpu" in sys.argv else mx.gpu)
    result = {
        "mask_syncs": mask_syncs(stock),
        "rope_offsets": rope_offsets(stock),
        "mask_decisions": mask_decisions(stock, owned),
        "image_mask_spread": image_mask_spread(stock),
        "lm_policy": lm_policy(stock),
        "unified_policy": unified_policy(stock_uni),
        "softcap": {"gmlx._vendor.mlx_vlm": softcap(stock.logit_softcap),
                    "gmlx._vendor.mlx_lm": softcap(stock_lm.logit_softcap)},
        "model_logits_dtype": model_logits_dtype(stock),
        "ple_batch_slice": ple_batch_slice(stock),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
