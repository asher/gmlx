"""The owned Gemma 4 model code against pristine stock, on the same weights.

A child process driver for tests/models/test_gemma4_owned_identity.py. It
imports gmlx, whose cache binding gives stock gemma4 the owned caches and
SDPA, and runs no loader install, so the stock classes run their mlx-vlm
0.7.3 bodies. Each tiny random config builds a stock tree and an owned
tree, loads the stock weights into the owned tree and runs the same calls
on both. The MoE reference gets gmlx's block routing on its mlx-vlm
SwitchGLUs, which is what gmlx runs today. Every logit passes the final
softcap, which the owned copy runs in float32, so the reference takes the
owned softcap too; test_gemma4_owned_softcap.py covers that change.

Prints one JSON object: case -> {check: true when identical}.
"""

from __future__ import annotations

import json
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

import gmlx  # noqa: F401  (the cache binding)
from gmlx.load.modules import install_switch_glu_block_routing
from gmlx.models.gemma4 import gemma4 as owned_g4
from gmlx.models.gemma4 import language as owned_lang
from gmlx.models.gemma4 import unified as owned_uni
from gmlx._vendor.mlx_vlm.models.gemma4 import gemma4 as stock_g4
from gmlx._vendor.mlx_vlm.models.gemma4 import language as stock_lang
from gmlx._vendor.mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
from gmlx._vendor.mlx_vlm.models.gemma4_unified import gemma4_unified as stock_uni
from gmlx._vendor.mlx_vlm.models.gemma4_unified.config import (
    ModelConfig as UnifiedModelConfig,
)

_BASE = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
    vocab_size_per_layer_input=128, num_key_value_heads=2,
    num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    sliding_window=32, sliding_window_pattern=3, tie_word_embeddings=True,
)
TEXT = {
    "dense": {},
    "kv_shared": {"num_kv_shared_layers": 2},
    "ple": {"num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8},
    "moe": {"enable_moe_block": True, "num_experts": 4, "top_k_experts": 2,
            "moe_intermediate_size": 32, "attention_k_eq_v": True,
            "num_global_key_value_heads": 1},
}
PROMPT = [3, 17, 42, 99, 7, 63, 5, 28, 11, 90, 4]
BLOCK = [5, 7, 9]


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict):
        return (isinstance(b, dict) and set(a) == set(b)
                and all(_same(a[k], b[k]) for k in a))
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        mx.array_equal(a, b).item())


def _trees(stock, owned) -> bool:
    ps = {k: (v.shape, v.dtype) for k, v in tree_flatten(stock.parameters())}
    po = {k: (v.shape, v.dtype) for k, v in tree_flatten(owned.parameters())}
    return ps == po


def _copy(stock, owned):
    owned.load_weights(list(tree_flatten(stock.parameters())), strict=True)


def _ids(toks, rows=1):
    return mx.array([list(toks)] * rows)


def _lm_pair(name):
    cfg = TextConfig(**{**_BASE, **TEXT[name]})
    mx.random.seed(13)
    stock = stock_lang.LanguageModel(cfg)
    mx.eval(stock.parameters())
    owned = owned_lang.LanguageModel(cfg)
    ok = _trees(stock, owned)
    _copy(stock, owned)
    mx.eval(owned.parameters())
    if name == "moe":
        install_switch_glu_block_routing(stock)
    return stock, owned, ok


def lm_case(name) -> dict:
    stock, owned, trees = _lm_pair(name)
    out = {"trees": trees}

    def both(fn):
        return fn(stock), fn(owned)

    # prefill, then greedy decode, then a verify-width block with the MTP
    # capture and sinks, then a rollback and one more step
    caches = {id(stock): stock.make_cache(), id(owned): owned.make_cache()}

    def call(m, toks, **kw):
        return m(_ids(toks), cache=caches[id(m)], **kw)

    s, o = both(lambda m: call(m, PROMPT).logits)
    out["prefill"] = _same(s, o)
    tok = int(mx.argmax(s[0, -1]).item())
    decode = True
    for _ in range(3):
        s, o = both(lambda m: call(m, [tok]).logits)
        decode &= _same(s, o)
        tok = int(mx.argmax(s[0, -1]).item())
    out["decode"] = decode

    def verify(m):
        r = call(m, BLOCK, return_hidden=True, return_shared_kv=True)
        return r.logits, r.hidden_states, r.shared_kv_states

    s, o = both(verify)
    out["verify_capture"] = _same(s, o)
    s, o = both(lambda m: m.rollback_speculative_cache(
        caches[id(m)], None, 0, len(BLOCK)))
    out["rollback"] = s == o
    s, o = both(lambda m: call(m, [BLOCK[0]]).logits)
    out["after_rollback"] = _same(s, o)

    s, o = both(lambda m: m(_ids(PROMPT), capture_layer_ids=[0, 3]))
    out["capture_layer_ids"] = _same(s.hidden_states, o.hidden_states) and \
        _same(s.logits, o.logits)

    def sinks(m):
        hid, kv = [], {}
        r = m(_ids(PROMPT), hidden_sink=hid, shared_kv_sink=kv,
              return_hidden=True)
        return r.logits, hid, kv

    s, o = both(sinks)
    out["caller_sinks"] = _same(s, o)
    s, o = both(lambda m: m(_ids(PROMPT), logits_to_keep=1).logits)
    out["logits_to_keep"] = _same(s, o)
    s, o = both(lambda m: m.model(_ids(PROMPT), skip_final_norm=True))
    out["skip_final_norm"] = _same(s, o)
    s, o = both(lambda m: m.speculative_logits_from_hidden(
        m(_ids(PROMPT), return_hidden=True).hidden_states[-1]))
    out["speculative_logits_from_hidden"] = _same(s, o)

    # two rows through one cache, then a decode step
    def rows(m):
        c = m.make_cache()
        a = m(_ids(PROMPT, 2), cache=c).logits
        return a, m(_ids([9], 2), cache=c).logits

    s, o = both(rows)
    out["two_rows"] = _same(s, o)

    if name == "moe":
        # verify widths 2 to 8 after a prefill, and one decode step
        widths = {}
        for t in range(1, 9):
            def block(m, t=t):
                c = m.make_cache()
                m(_ids(PROMPT), cache=c)
                return m(_ids(list(range(20, 20 + t))), cache=c).logits

            s, o = both(block)
            widths[str(t)] = _same(s, o)
        out["moe_widths"] = widths
        out["switch_glu"] = sorted({
            f"{type(layer.experts.switch_glu).__module__}."
            f"{type(layer.experts.switch_glu).__name__}"
            for layer in owned.model.layers})
    return out


def _container(kind):
    if kind == "gemma4":
        cfg = ModelConfig(text_config=TextConfig(**{**_BASE, **TEXT["ple"]}),
                          vision_config=VisionConfig())
        return cfg, stock_g4.Model, owned_g4.Model
    text = {**_BASE, "model_type": "gemma4_unified_text"}
    cfg = UnifiedModelConfig.from_dict({"model_type": "gemma4_unified",
                                        "text_config": text,
                                        "vision_config": {}})
    return cfg, stock_uni.Model, owned_uni.Model


def container_case(kind) -> dict:
    cfg, stock_cls, owned_cls = _container(kind)
    mx.random.seed(13)
    stock = stock_cls(cfg)
    mx.eval(stock.language_model.parameters())
    owned = owned_cls(cfg)
    out = {"trees": _trees(stock, owned)}
    _copy(stock, owned)
    mx.eval(owned.language_model.parameters())
    out["language_model_class"] = type(owned.language_model).__module__
    caches = {id(stock): stock.make_cache(), id(owned): owned.make_cache()}

    def call(m, toks):
        return m(_ids(toks), cache=caches[id(m)]).logits

    s, o = call(stock, PROMPT), call(owned, PROMPT)
    out["prefill"] = _same(s, o)
    tok = int(mx.argmax(s[0, -1]).item())
    decode = True
    for _ in range(2):
        s, o = call(stock, [tok]), call(owned, [tok])
        decode &= _same(s, o)
        tok = int(mx.argmax(s[0, -1]).item())
    out["decode"] = decode
    names = {k: mx.zeros((1,)) for k, _ in tree_flatten(
        stock.language_model.parameters())}
    names["model.language_model.layers.0.self_attn.rotary_emb.inv_freq"] = \
        mx.zeros((1,))
    out["sanitize"] = sorted(stock.sanitize(names)) == sorted(
        owned.sanitize(names))
    return out


def main():
    mx.set_default_device(mx.cpu if "--cpu" in sys.argv else mx.gpu)
    stock_lang.logit_softcap = owned_lang.logit_softcap
    result = {name: lm_case(name) for name in TEXT}
    for kind in ("gemma4", "gemma4_unified"):
        result[kind] = container_case(kind)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
