"""The K to V head order of a Qwen3.5 gated delta layer is the model's own.

With fewer K heads than V heads, an MLX checkpoint keeps its V heads
grouped (V head hv reads K head hv // (Hv / Hk)) and a GGUF keeps them
tiled (hv % Hk). The owned language model reads ``kv_head_layout`` from
its text config and passes it to every scan of
``gmlx.models.qwen35.gated_delta``, so one process holds a model of each
order.

The model cases run in child processes: a tiled model and a grouped model
with the same weights, built through the copied container in either
order. Each model gives the logits of a reference with one K head per V head
that holds the pairing in its weights, and the logits it gives as the
only model of its process. The scan cases call every entry of the module
with each order against inputs expanded to one K head per V head.
"""

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import gmlx.models.qwen35.gated_delta as vgd

_GPU = pytest.param("gpu", marks=pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")), reason="needs the Metal GPU"))
_DEVICES = (_GPU, "cpu")
_ORDERS = (("tiled", "grouped"), ("grouped", "tiled"))

# Model against its one-K-head-per-V-head reference. The two run the same
# scan on rows that hold equal values, and measured equal to the bit on
# the GPU and the CPU, on logits that reach 5.
REF_ATOL = 1e-5
# Model against the same model alone in its process: the same code on the
# same weights, also equal to the bit.
ALONE_ATOL = 1e-6
# Neither order may pass for the other: they measured 1.57 apart on the
# prefill and 0.47 on the decode.
ORDERS_DIFFER = 0.1

_CHILD = r"""
import json
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

order, device = json.loads(sys.argv[1]), sys.argv[2]
mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)

from gmlx._vendor.mlx_vlm.models.qwen3_5 import qwen3_5 as container
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import (
    ModelConfig, TextConfig, VisionConfig)

HK, HV, DK = 2, 6, 32


def text_raw(layout, hk):
    return dict(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=HV, linear_num_key_heads=hk,
        linear_key_head_dim=DK, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=2048, tie_word_embeddings=True, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.25},
        full_attention_interval=4, kv_head_layout=layout)


def build(layout, hk=HK):
    text = TextConfig.from_dict(text_raw(layout, hk))
    cfg = ModelConfig(
        text_config=text,
        vision_config=VisionConfig(
            depth=1, hidden_size=32, intermediate_size=64, out_hidden_size=64,
            num_heads=2, patch_size=14, num_position_embeddings=16,
            deepstack_visual_indexes=[]),
        model_type="qwen3_5", vocab_size=128, image_token_id=124,
        video_token_id=125, vision_start_token_id=126, vision_end_token_id=127)
    mx.random.seed(0)
    # The class the copied container builds for an MLX checkpoint.
    lm = container.LanguageModel(text, cfg)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def reference(lm, layout):
    # One K head per V head, K head j being the model's K head heads[j].
    heads = [h % HK if layout == "tiled" else h // (HV // HK)
             for h in range(HV)]
    kd, vd = HK * DK, HV * 32
    qk = np.concatenate([np.arange(h * DK, (h + 1) * DK) for h in heads])
    idx = mx.array(np.concatenate([qk, kd + qk, 2 * kd + np.arange(vd)]))
    ref = build("grouped", hk=HV)
    weights = []
    for name, value in tree_flatten(lm.parameters()):
        if name.endswith(("linear_attn.in_proj_qkv.weight",
                          "linear_attn.conv1d.weight")):
            value = value[idx]
        weights.append((name, value))
    ref.load_weights(weights, strict=True)
    mx.eval(ref.parameters())
    return ref


def logits(lm, ids):
    cache = lm.make_cache()
    prefill = lm(ids[:, :5], cache=cache).logits
    mx.eval(prefill)
    decode = lm(ids[:, 5:6], cache=cache).logits
    mx.eval(decode)
    return {"prefill": np.array(prefill.astype(mx.float32)).tolist(),
            "decode": np.array(decode.astype(mx.float32)).tolist()}


import gmlx.models.qwen35.language as language

# The head order of each call of the scan that recomputes a rolled-back
# state.
recomputes = []
_state_update = language.gated_delta_state_update


def _counted_state_update(*args, **kw):
    recomputes.append(bool(kw.get("tiled")))
    return _state_update(*args, **kw)


language.gated_delta_state_update = _counted_state_update


def plain(lm, ids):
    cache = lm.make_cache()
    mx.eval(lm(ids[:, :5], cache=cache).logits)
    for i in (5, 6, 7):
        y = lm(ids[:, i:i + 1], cache=cache).logits.astype(mx.float32)
        mx.eval(y)
    return np.array(y).tolist()


def rollback(lm, ids, recompute):
    # Prefill 5 tokens, verify a block of 4, accept 1, then feed one more.
    # With ``recompute`` the verify hands the rollback no captured states
    # (entry 11 of each layer's record), so the rollback recomputes the
    # state from the verify inputs.
    cache = lm.make_cache()
    mx.eval(lm(ids[:, :5], cache=cache).logits)
    hidden, _kv, gdn = lm.speculative_verify_hidden(ids[:, 5:9], cache)
    mx.eval(hidden)
    if recompute:
        gdn = [tuple(s[:11]) + (None,) + tuple(s[12:]) for s in gdn]
    before = len(recomputes)
    lm.rollback_speculative_cache(cache, gdn, 1, 4)
    y = lm(ids[:, 7:8], cache=cache).logits.astype(mx.float32)
    mx.eval(y)
    return np.array(y).tolist(), recomputes[before:]


ids = mx.array(np.random.default_rng(0).integers(1, 120, size=(1, 6)))
ids9 = mx.array(np.random.default_rng(1).integers(1, 120, size=(1, 9)))
models = {layout: build(layout) for layout in order}
out = {}
for layout, lm in models.items():
    captured, captured_calls = rollback(lm, ids9, False)
    recomputed, recompute_calls = rollback(lm, ids9, True)
    out[layout] = {"model": logits(lm, ids),
                   "reference": logits(reference(lm, layout), ids),
                   "class": type(lm).__module__,
                   "rollback": {"plain": plain(lm, ids9),
                                "captured": captured,
                                "captured_calls": captured_calls,
                                "recomputed": recomputed,
                                "recompute_calls": recompute_calls}}
print(json.dumps(out))
"""

_results: dict = {}


def _run(order, device):
    key = (tuple(order), device)
    if key not in _results:
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, json.dumps(list(order)), device],
            capture_output=True, text=True, timeout=900,
            env=dict(os.environ))
        assert r.returncode == 0, r.stderr[-4000:]
        _results[key] = json.loads(r.stdout.strip().splitlines()[-1])
    return _results[key]


def _gap(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).max())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("order", _ORDERS, ids=("tiled_first", "grouped_first"))
def test_each_model_keeps_its_own_head_order(order, device):
    both = _run(order, device)
    for layout in order:
        assert both[layout]["class"] == "gmlx.models.qwen35.language"
        alone = _run((layout,), device)[layout]
        for phase in ("prefill", "decode"):
            got = both[layout]["model"][phase]
            ref = _gap(got, both[layout]["reference"][phase])
            assert ref < REF_ATOL, (
                f"{layout} model, {phase} on {device}, loaded "
                f"{' then '.join(order)}: {ref} from its reference")
            same = _gap(got, alone["model"][phase])
            assert same <= ALONE_ATOL, (
                f"{layout} model, {phase} on {device}, loaded "
                f"{' then '.join(order)}: {same} from the model alone")
    for phase in ("prefill", "decode"):
        apart = _gap(both["tiled"]["model"][phase],
                     both["grouped"]["model"][phase])
        assert apart > ORDERS_DIFFER, (phase, apart)


# A rollback against the plain decode of the same tokens: the measured
# gap is under 1e-6 on logits that reach 5.
ROLLBACK_ATOL = 1e-4


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("layout", ("tiled", "grouped"))
def test_rollback_that_recomputes_the_state_keeps_the_head_order(layout, device):
    """A rollback whose verify captured no states recomputes the recurrent
    state from the verify inputs, in the head order of its model."""
    other = "grouped" if layout == "tiled" else "tiled"
    got = _run((layout,), device)[layout]["rollback"]
    apart = _run((other,), device)[other]["rollback"]["plain"]
    # The rollback with captured states never reaches the scan, and the
    # one without them reaches it once.
    assert got["captured_calls"] == []
    assert got["recompute_calls"] == [layout == "tiled"]
    for path in ("captured", "recomputed"):
        gap = _gap(got[path], got["plain"])
        assert gap < ROLLBACK_ATOL, (
            f"{layout} model on {device}, {path} rollback: {gap} from the "
            "plain decode")
        assert _gap(got[path], apart) > ORDERS_DIFFER, (layout, device, path)


# The scans, one entry at a time.

B, HK, HV, DK, DV = 2, 2, 6, 32, 32


@pytest.fixture(params=_DEVICES)
def on_device(request):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu if request.param == "cpu" else mx.gpu)
    yield request.param
    mx.set_default_device(prev)


def _arr(rng, *shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def _inputs(T, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        q=_arr(rng, B, T, HK, DK) * 0.2, k=_arr(rng, B, T, HK, DK) * 0.2,
        v=_arr(rng, B, T, HV, DV), a=_arr(rng, B, T, HV),
        b=_arr(rng, B, T, HV), A_log=_arr(rng, HV), dt_bias=_arr(rng, HV),
        state=_arr(rng, B, HV, DV, DK) * 0.1)


def _one_k_head_per_v_head(x, tiled):
    heads = [h % HK if tiled else h // (HV // HK) for h in range(HV)]
    return x[:, :, mx.array(heads)]


def _mask(T):
    m = np.ones((B, T), dtype=bool)
    m[1, 0] = False
    return mx.array(m)


def _close(got, want, atol=1e-4):
    """A kernel returns its outputs as a list, the ops as a tuple."""
    got = list(got) if isinstance(got, (tuple, list)) else [got]
    want = list(want) if isinstance(want, (tuple, list)) else [want]
    assert len(got) == len(want)
    mx.eval(got, want)
    return max(_gap(np.array(g), np.array(w))
               for g, w in zip(got, want)) < atol


def _update(kw, tiled, **over):
    return vgd.gated_delta_update(
        kw["q"], kw["k"], kw["v"], kw["a"], kw["b"], kw["A_log"],
        kw["dt_bias"], kw["state"], over.pop("mask", None), tiled=tiled,
        **over)


def _with_states(kw, tiled, **over):
    return vgd.gated_delta_update_with_states(
        kw["q"], kw["k"], kw["v"], kw["a"], kw["b"], kw["A_log"],
        kw["dt_bias"], kw["state"], over.pop("mask", None), tiled=tiled,
        **over)


def _state_update(kw, tiled, **over):
    steps = mx.array([kw["k"].shape[1], 1], dtype=mx.int32)
    return vgd.gated_delta_state_update(
        kw["k"], kw["v"], kw["a"], kw["b"], kw["A_log"], kw["dt_bias"],
        kw["state"], steps, over.pop("mask", None), tiled=tiled, **over)


_ENTRIES = {"update": _update, "with_states": _with_states,
            "state_update": _state_update}


@pytest.mark.parametrize("masked", (False, True), ids=("nomask", "mask"))
@pytest.mark.parametrize("use_kernel", (True, False), ids=("kernel", "ops"))
@pytest.mark.parametrize("T", (1, 4))
@pytest.mark.parametrize("tiled", (False, True), ids=("grouped", "tiled"))
@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_every_scan_pairs_heads_in_the_order_it_is_given(
        on_device, entry, tiled, T, use_kernel, masked):
    """Each entry, with ``tiled`` set either way, equals itself on inputs
    that hold the pairing: K head j of the expanded input is K head
    ``heads[j]``. The other order gives a different answer."""
    kw = _inputs(T)
    mask = _mask(T) if masked else None
    fn = _ENTRIES[entry]
    got = fn(kw, tiled, mask=mask, use_kernel=use_kernel)
    wide = dict(kw, q=_one_k_head_per_v_head(kw["q"], tiled),
                k=_one_k_head_per_v_head(kw["k"], tiled))
    for flag in (False, True):
        assert _close(got, fn(wide, flag, mask=mask, use_kernel=use_kernel))
    assert not _close(got, fn(kw, not tiled, mask=mask,
                              use_kernel=use_kernel), atol=1e-2)


@pytest.mark.parametrize("tiled", (False, True), ids=("grouped", "tiled"))
def test_chunked_prefill_scan_pairs_heads_in_the_order_it_is_given(
        on_device, tiled, monkeypatch):
    """A prefill off the GPU, or with the kernels off, runs
    ``gated_delta_chunked``. It takes the head order from the update, and
    agrees with the step-by-step ops in that order. The chunked form
    measured 2e-7 from the ops on the CPU and 2.2e-3 on the GPU, in either
    order, and the other order is 1.2 away."""
    atol = 1e-4 if on_device == "cpu" else 1e-2
    kw = _inputs(7)
    seen = []
    chunked = vgd.gated_delta_chunked
    monkeypatch.setattr(
        vgd, "gated_delta_chunked",
        lambda *a, **k: seen.append(k.get("tiled")) or chunked(*a, **k))
    got = _update(kw, tiled, use_kernel=False)
    assert seen == [tiled]
    g, beta = vgd._compute_g_beta(kw["A_log"], kw["a"], kw["b"], kw["dt_bias"])
    steps = vgd.gated_delta_ops(kw["q"], kw["k"], kw["v"], g, beta,
                                kw["state"], None, tiled=tiled)
    assert _close(got, steps, atol=atol)
    wide = dict(kw, q=_one_k_head_per_v_head(kw["q"], tiled),
                k=_one_k_head_per_v_head(kw["k"], tiled))
    assert _close(got, _update(wide, False, use_kernel=False), atol=atol)
    assert not _close(got, _update(kw, not tiled, use_kernel=False),
                      atol=0.1)


@pytest.mark.skipif(bool(os.environ.get("KQUANT_FORCE_CPU")),
                    reason="needs the Metal GPU")
@pytest.mark.parametrize("masked", (False, True), ids=("nomask", "mask"))
@pytest.mark.parametrize("tiled", (False, True), ids=("grouped", "tiled"))
def test_prefill_kernel_with_a_gate_per_key_dim_pairs_heads(tiled, masked):
    """The kernel variant with a gate per key dimension, which no update
    entry reaches, against the ops in the same order. At a masked step the
    kernel writes zero and the ops an output nothing reads, so the outputs
    are compared at the valid steps."""
    kw = _inputs(4)
    rng = np.random.default_rng(1)
    g = mx.sigmoid(_arr(rng, B, 4, HV, DK))
    beta = mx.sigmoid(kw["b"])
    mask = _mask(4) if masked else None

    def valid(out):
        y, state = out
        return [y if mask is None else mx.where(mask[..., None, None], y, 0),
                state]

    got = valid(vgd.gated_delta_kernel(kw["q"], kw["k"], kw["v"], g, beta,
                                       kw["state"], mask, tiled=tiled))
    want = valid(vgd.gated_delta_ops(kw["q"], kw["k"], kw["v"], g, beta,
                                     kw["state"], mask, tiled=tiled))
    assert _close(got, want)
    other = valid(vgd.gated_delta_ops(kw["q"], kw["k"], kw["v"], g, beta,
                                      kw["state"], mask, tiled=not tiled))
    assert not _close(got, other, atol=1e-2)


def test_no_installer_rebinds_the_scan_module():
    """The head order is a call argument. Nothing in gmlx assigns a scan
    of the module, which the copied Qwen3.5 path resolves to and every
    model in the process shares."""
    import gmlx.upstream.gdn_patches as gdn_patches

    assert not hasattr(gdn_patches, "_patch_mlxvlm_gated_delta_tiled_v")
    for name in ("gated_delta_ops", "gated_delta_kernel",
                 "gated_delta_chunked", "gated_delta_update",
                 "_gated_delta_with_states_ops", "_gated_delta_state_ops"):
        assert getattr(vgd, name).__module__ == vgd.__name__, name
    for name in ("_gated_delta_with_states_kernel",
                 "_gated_delta_with_states_kernel_masked",
                 "_gated_delta_state_kernel",
                 "_gated_delta_state_kernel_masked"):
        assert (getattr(vgd, name) is None) == (not mx.metal.is_available())
