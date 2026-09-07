"""DeepSeek-V4-Flash-Vision-Exp: image blocks, in-span attention masks, the
image router bias, the row patch, and the VLM container (CPU, tiny config).

Reference for the masks is the checkpoint's own inference code
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, HF commit 6821d6ad,
inference/model.py get_window_topk_idxs_visible), ported to numpy in
``_ref_visible``; ``build_image_block`` is ported in image_block.py."""

import os
import sys

import numpy as np
import mlx.core as mx
import mlx.utils as mu
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "load"))

import gmlx.models.deepseek_v4.model as v4
from gmlx.load.config_synth import synthesize_config
from gmlx.models.deepseek_v4 import image_block as ib
from gmlx.models.deepseek_v4.mtp import (
    DeepseekV4MTPConfig,
    DeepseekV4MTPDrafter,
    DeepseekV4SpecLM,
)
from gmlx.models.deepseek_v4.vision import (
    DeepseekV4Aligner,
    VisionConfig,
    vision_rope_tables,
)
from gmlx.models.deepseek_v4.vlm_model import Model as VLModel
from gmlx.models.deepseek_v4.vlm_model import ModelConfig
from mlx_lm.models.base import create_attention_mask

from test_config_synth import _DEEPSEEK4_SHAPES, _deepseek4_meta
from test_deepseek_v4_mtp import _randomize_zero_params

N_LAYERS = 3


def _cfg(vl=True):
    v4.ensure_registered()
    shapes = dict(_DEEPSEEK4_SHAPES)
    if vl:
        for i in range(N_LAYERS):
            shapes[f"blk.{i}.exp_probs_b_vl.bias"] = [8]  # expert_count
    cfg = synthesize_config(_deepseek4_meta(), tensor_shapes=shapes)
    assert cfg["vision_router_bias"] is vl
    return cfg


def _text_model(cfg=None):
    cfg = cfg or _cfg()
    lm = v4.Model(v4.ModelArgs.from_dict(cfg))
    mx.eval(lm.parameters())
    _randomize_zero_params(lm)
    return lm, lm.args


def _ref_visible(ids, vocab, window, max_img=384):
    """numpy port of get_window_topk_idxs_visible (visibility only)."""
    ids = np.asarray(ids)
    n = len(ids)
    idx = np.arange(n)
    is_start = ids == vocab + ib.IMAGE_START
    is_end = ids == vocab + ib.IMAGE_END
    valid = (np.cumsum(is_start) > np.cumsum(is_end)) | is_end
    starts = np.maximum.accumulate(np.where(is_start, idx, 0))
    left = np.minimum((idx - starts) * valid, max_img - 1)
    ends = np.minimum.accumulate(np.where(is_end, idx, n)[::-1])[::-1]
    right = np.minimum((ends - idx) * valid, max_img)
    left_add = np.maximum(left - (window - 1), 0)
    lo = np.maximum(idx - (window - 1) - left_add, 0)
    hi = idx + right
    k = idx[None, :]
    return (k >= lo[:, None]) & (k <= hi[:, None])


def _prompt(vocab, blocks_at, n_h=2, n_w=8, total=None, seed=0):
    """Token ids with image blocks whose first lead pad sits at each of
    ``blocks_at``; returns (ids, image_spans)."""
    rng = np.random.default_rng(seed)
    ids, spans = [], []
    for start in blocks_at:
        while len(ids) < start:
            ids.append(int(rng.integers(1, vocab)))
        types, _ = ib.build_image_block(n_h, n_w, start)
        spans.append([start, start + len(types)])
        ids.extend(int(vocab + t) for t in types)
    while len(ids) < (total or len(ids) + 5):
        ids.append(int(rng.integers(1, vocab)))
    return ids, spans


def _chunk_masks(lm, args, ids, spans, off, L):
    """Masks exactly as DeepseekV4Model builds them for the chunk
    [off, off+L) after prefilling ids[:off] (text) into a real cache."""
    cache = lm.make_cache()
    if off:
        lm(mx.array([ids[:off]]), cache=cache)
        mx.eval([c.state for c in _leaf_caches(cache)])
    mask_cache = cache[0]
    while isinstance(mask_cache, (list, tuple)) or hasattr(mask_cache, "caches"):
        mask_cache = (mask_cache[0] if isinstance(mask_cache, (list, tuple))
                      else mask_cache.caches[0])
    h = mx.zeros((1, L, args.hidden_size))
    mask = create_attention_mask(
        h, mask_cache, window_size=args.sliding_window, return_array=True)
    blocks = v4.image_chunk_blocks(spans, off, L)
    return v4._image_chunk_masks(mask, blocks, 1, L), blocks


def _leaf_caches(cache):
    out = []
    for c in cache:
        if hasattr(c, "caches"):
            out.extend(c.caches)
        elif isinstance(c, (list, tuple)):
            out.extend(c)
        else:
            out.append(c)
    return [c for c in out if hasattr(c, "state")]


@pytest.mark.parametrize("off,L,blocks_at,n_hw", [
    (0, 40, [4], (2, 8)),          # span at the chunk start, offset 0
    (20, 48, [24], (2, 8)),        # offset >= window, block longer than it
    (20, 80, [23, 60], (2, 8)),    # two blocks, odd/even lead pads
    (0, 30, [3], (1, 3)),          # block shorter than the window
])
def test_in_span_mask_matches_reference(off, L, blocks_at, n_hw):
    lm, args = _text_model()
    W = args.sliding_window
    vocab = args.vocab_size
    ids, spans = _prompt(vocab, blocks_at, *n_hw, total=off + L)
    assert all(e - s > W for s, e in spans) or n_hw == (1, 3)
    (merged, image_mask, rows), blocks = _chunk_masks(
        lm, args, ids, spans, off, L)
    merged = np.array(merged)
    assert merged.ndim == 2 and merged.shape[0] == L  # 2-D [L, S], as consumed
    S = merged.shape[-1]
    ref = _ref_visible(ids[:off + L], vocab, W)
    for i in range(L):
        for j in range(S):
            k = off + L - S + j
            if k < 0:
                assert not merged[i, j]
                continue
            assert merged[i, j] == ref[off + i, k], (i, k)
    assert np.array_equal(
        np.array(image_mask)[0], np.array(ids[off:off + L]) >= vocab)
    # widened rows = START..END of every block, lead pads excluded
    exp_rows = []
    for (s, e), (a, b, lead) in zip(spans, blocks):
        exp_rows.append((a + lead, b))
    assert rows == exp_rows


def test_no_spans_is_the_plain_window_mask():
    lm, args = _text_model()
    L = 24
    cache = lm.make_cache()
    mask_cache = cache[0]
    while isinstance(mask_cache, (list, tuple)) or hasattr(mask_cache, "caches"):
        mask_cache = (mask_cache[0] if isinstance(mask_cache, (list, tuple))
                      else mask_cache.caches[0])
    mask = create_attention_mask(
        mx.zeros((1, L, args.hidden_size)), mask_cache,
        window_size=args.sliding_window, return_array=True)
    merged, image_mask, rows = v4._image_chunk_masks(mask, [], 1, L)
    assert np.array_equal(np.array(merged), np.array(mask))
    assert not bool(mx.any(image_mask)) and rows == []


def test_block_cut_by_chunk_raises():
    lm, args = _text_model()
    ids, spans = _prompt(args.vocab_size, [4], total=40)
    with pytest.raises(ValueError, match="cut by the prefill chunk"):
        lm(mx.array([ids[:10]]), cache=lm.make_cache(), image_spans=spans)
    # blocks outside the chunk are ignored
    assert v4.image_chunk_blocks(spans, 0, 4) == []
    assert v4.image_chunk_blocks(spans, spans[0][1], 5) == []


def _gate_reference(gate, x, ids, image):
    logits = mx.matmul(x, gate.weight.T)
    scores = np.array(v4._score_func(logits.astype(mx.float32),
                                     gate.scoring_func))
    out_inds, out_w = [], []
    for t in range(x.shape[1]):
        s = scores[0, t]
        if image[t]:
            top = np.argsort(-(s + np.array(gate.e_score_correction_bias_vl)))
        elif gate.hash:
            top = np.array(gate.tid2eid)[int(ids[0, t])]
        else:
            top = np.argsort(-(s + np.array(gate.e_score_correction_bias)))
        top = top[:gate.top_k]
        w = s[top]
        if gate.norm_topk_prob:
            w = w / w.sum()
        out_inds.append(sorted(top.tolist()))
        out_w.append(dict(zip(top.tolist(), (w * gate.routed_scaling_factor).tolist())))
    return out_inds, out_w


@pytest.mark.parametrize("layer", [0, N_LAYERS - 1])
def test_vl_gate_matches_numpy(layer):
    cfg = _cfg()
    args = v4.ModelArgs.from_dict(cfg)
    gate = v4.MoEGate(args, layer)
    mx.eval(gate.parameters())
    _randomize_zero_params(gate)
    if gate.hash:
        gate.tid2eid = mx.random.randint(
            0, args.n_routed_experts, (args.vocab_size, gate.top_k)
        ).astype(mx.int32)
    T = 6
    x = mx.random.normal((1, T, args.hidden_size))
    ids = mx.random.randint(1, args.vocab_size, (1, T))
    image = [False, False, True, True, True, False]
    inds, w = gate(x, ids, mx.array([image]))
    inds0, w0 = gate(x, ids, None)
    mx.eval(inds, w, inds0, w0)
    ref_inds, ref_w = _gate_reference(gate, x, ids, image)
    for t in range(T):
        got = inds[0, t].tolist()
        assert sorted(got) == ref_inds[t], (t, got, ref_inds[t])
        for e, wt in zip(got, w[0, t].tolist()):
            assert abs(wt - ref_w[t][e]) < 1e-4, (t, e, wt, ref_w[t][e])
        if not image[t]:
            assert sorted(inds0[0, t].tolist()) == sorted(got)
            assert np.allclose(np.array(w0[0, t]), np.array(w[0, t]), atol=1e-6)


def test_gate_without_vl_bias_rejects_image_mask():
    args = v4.ModelArgs.from_dict(_cfg(vl=False))
    gate = v4.MoEGate(args, N_LAYERS - 1)
    assert not hasattr(gate, "e_score_correction_bias_vl")
    with pytest.raises(ValueError, match="exp_probs_b_vl"):
        gate(mx.zeros((1, 2, args.hidden_size)), mx.array([[1, 2]]),
             mx.array([[False, True]]))


def _fake_window_kernel(module, q, kv, pooled, sinks, offset, ratio):
    """Stand-in for the kq window kernel: plain sliding-window sdpa (no
    span visibility), so the row patch has something to fix up."""
    if pooled is not None:
        return None
    B, H, L, D = q.shape
    S = kv.shape[2]
    qi = int(offset) + np.arange(L)[:, None]
    kj = int(offset) + L - S + np.arange(S)[None, :]
    W = module.config.sliding_window
    plain = mx.array((kj <= qi) & (kj > qi - W))
    return v4.scaled_dot_product_attention(
        q, kv, kv, cache=None, scale=module.scale, mask=plain, sinks=sinks)


@pytest.mark.parametrize("blocks_at", [[4], [3, 40]])
def test_row_patch_matches_full_fallback(monkeypatch, blocks_at):
    lm, args = _text_model()
    ids, spans = _prompt(args.vocab_size, blocks_at, total=80)
    monkeypatch.setattr(v4, "_kernel_window_attention", _fake_window_kernel)
    monkeypatch.setattr(v4, "_dsa_probe", lambda path: path == "window")
    monkeypatch.setattr(v4, "_SPAN_FULL_FALLBACK", False)
    patched = lm(mx.array([ids]), cache=lm.make_cache(), image_spans=spans)
    monkeypatch.setattr(v4, "_SPAN_FULL_FALLBACK", True)
    full = lm(mx.array([ids]), cache=lm.make_cache(), image_spans=spans)
    mx.eval(patched, full)
    assert np.isfinite(np.array(patched)).all()
    assert np.allclose(np.array(patched), np.array(full), atol=2e-4, rtol=1e-3)
    # the plain kernel result alone is wrong on the span rows: drop the
    # patch and the logits move
    monkeypatch.setattr(v4, "_patch_span_rows",
                        lambda out, image_rows, recompute: out)
    monkeypatch.setattr(v4, "_SPAN_FULL_FALLBACK", False)
    unpatched = lm(mx.array([ids]), cache=lm.make_cache(), image_spans=spans)
    assert not np.allclose(np.array(unpatched), np.array(full), atol=2e-4)


def test_vl_bias_params_only_on_trunk_layers():
    cfg = _cfg()
    lm = DeepseekV4SpecLM(v4.ModelArgs.from_dict(cfg))
    keys = [k for k, _ in mu.tree_flatten(lm.parameters())
            if k.endswith("e_score_correction_bias_vl")]
    assert len(keys) == N_LAYERS
    off = DeepseekV4SpecLM(v4.ModelArgs.from_dict(_cfg(vl=False)))
    assert not [k for k, _ in mu.tree_flatten(off.parameters())
                if k.endswith("_vl")]
    args = v4.ModelArgs.from_dict(cfg)
    args.compress_ratios = list(args.compress_ratios) + [0]
    drafter = DeepseekV4MTPDrafter(DeepseekV4MTPConfig(text=args, block_size=2))
    assert not [k for k, _ in mu.tree_flatten(drafter.parameters())
                if k.endswith("_vl")]


def test_speclm_inputs_embeds_pass_through():
    lm = DeepseekV4SpecLM(v4.ModelArgs.from_dict(_cfg()))
    mx.eval(lm.parameters())
    _randomize_zero_params(lm)
    ids = mx.array([[1, 2, 3, 4, 5]])
    a = lm(ids).logits
    b = lm(ids, inputs_embeds=lm.model.embed_tokens(ids)).logits
    c = lm(ids, inputs_embeds=mx.zeros_like(lm.model.embed_tokens(ids))).logits
    mx.eval(a, b, c)
    assert np.allclose(np.array(a), np.array(b), atol=1e-5)
    assert not np.allclose(np.array(a), np.array(c), atol=1e-3)


def _container():
    cfg = _cfg()
    tc = v4.ModelArgs.from_dict(cfg)
    vc = VisionConfig(depth=1, hidden_size=16, num_heads=2,
                      intermediate_size=16, patch_size=2, in_channels=3,
                      out_hidden_size=tc.hidden_size, downsample_ratio=3)
    model = VLModel(ModelConfig(text_config=tc, vision_config=vc,
                                vocab_size=tc.vocab_size))
    mx.eval(model.parameters())
    _randomize_zero_params(model)
    return model, tc


def test_container_forward_with_image():
    model, tc = _container()
    vocab = tc.vocab_size
    assert model.language_model.config.media_token_ids == [
        vocab + t for t in range(5)]
    n_vit = 6                       # 6x6 patches -> 2x2 aligner grid
    ids, spans = _prompt(vocab, [6], n_h=2, n_w=2, total=30)
    pv = mx.random.normal((n_vit * n_vit, 3 * 2 * 2))
    meta = [[n_vit, n_vit, 2, 2]]
    feats = model.get_input_embeddings(
        mx.array([ids]), pv, image_meta=meta, image_spans=spans)
    emb = feats.inputs_embeds
    mx.eval(emb)
    assert emb.shape == (1, len(ids), tc.hidden_size)
    assert np.isfinite(np.array(emb)).all()
    for t, vec in ((ib.IMAGE_START, model.image_start),
                   (ib.IMAGE_PAD, model.image_pad),
                   (ib.IMAGE_NEW_LINE, model.image_newline),
                   (ib.IMAGE_END, model.image_end)):
        pos = [i for i, x in enumerate(ids) if x == vocab + t]
        assert pos
        assert np.allclose(np.array(emb[0, pos]), np.array(vec)[None], atol=1e-6)
    rows = model.vision_tower(pv, n_vit, n_vit)
    _, perm = ib.build_image_block(2, 2, 6)
    img_pos = [i for i, x in enumerate(ids) if x == vocab + ib.IMAGE]
    assert len(img_pos) == 4
    assert np.allclose(np.array(emb[0, img_pos]),
                       np.array(rows[mx.array(perm)]), atol=1e-5)
    txt = [i for i, x in enumerate(ids) if x < vocab]
    ref = model.language_model.model.embed_tokens(mx.array([ids[i] for i in txt]))
    assert np.allclose(np.array(emb[0, txt]), np.array(ref), atol=1e-6)
    out = model(mx.array([ids]), pv, cache=model.make_cache(),
                image_meta=meta, image_spans=spans)
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    assert logits.shape == (1, len(ids), vocab)
    assert np.isfinite(np.array(logits)).all()
    # slot / row count mismatch and missing pixels are errors
    with pytest.raises(ValueError, match="IMAGE slots"):
        model.get_input_embeddings(
            mx.array([ids + [vocab + ib.IMAGE]]), pv, image_meta=meta)
    with pytest.raises(ValueError, match="pixel_values"):
        model.get_input_embeddings(mx.array([ids]), None)
    # text prompts embed plainly
    plain = model.get_input_embeddings(mx.array([ids[:6]]), None).inputs_embeds
    assert plain.shape == (1, 6, tc.hidden_size)


def test_chunked_prefill_policy_declines_when_a_block_would_be_cut():
    from gmlx.gen.media_spans import stamp_prefill_step

    model, tc = _container()
    lm = model.language_model
    ids, spans = _prompt(tc.vocab_size, [40], total=140)
    embeds = mx.zeros((1, len(ids), tc.hidden_size))
    stamp_prefill_step(model, 50)
    assert lm.chunked_prefill_policy(
        inputs_embeds=embeds, prefill_kwargs={"image_spans": spans}) is False
    stamp_prefill_step(model, 200)
    assert lm.chunked_prefill_policy(
        inputs_embeds=embeds, prefill_kwargs={"image_spans": spans}) is True
    stamp_prefill_step(model, 50)
    assert lm.chunked_prefill_policy(inputs_embeds=embeds) is True


def test_block_length_and_lead_pads():
    for start in range(8):
        for n_h, n_w in ((1, 1), (2, 5), (3, 7), (16, 24)):
            types, perm = ib.build_image_block(n_h, n_w, start)
            assert len(types) == ib.block_length(n_h, n_w, start)
            assert types[0] == ib.IMAGE_PAD or ib.lead_pads(start) == 0
            assert int((np.asarray(types) == ib.IMAGE).sum()) == n_h * n_w
            assert sorted(perm) == list(range(n_h * n_w))
            assert types[-1] == ib.IMAGE_END
            assert (start + ib.lead_pads(start)) % 4 == 3


def test_resize_geometry_bounds():
    vc = VisionConfig()
    for w, h in ((1024, 768), (300, 300), (4000, 200), (100, 3000)):
        best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, stretch = (
            ib.resize_geometry(w, h, patch_size=vc.patch_size,
                               downsample_ratio=vc.downsample_ratio,
                               max_n_token=vc.max_n_token,
                               max_wh_ratio=vc.max_wh_ratio,
                               min_pixels=vc.min_pixels))
        assert best_w % vc.patch_size == 0 and best_h % vc.patch_size == 0
        assert n_vit_w == best_w // vc.patch_size
        assert n_vit_h == best_h // vc.patch_size
        assert n_llm_w == -(-n_vit_w // vc.downsample_ratio)
        assert n_llm_h == -(-n_vit_h // vc.downsample_ratio)
        assert ib.block_length(n_llm_h, n_llm_w, 0) <= vc.max_n_token
        assert best_w <= vc.max_wh_ratio * best_h or stretch


def test_aligner_unfold_matches_numpy():
    vc = VisionConfig(hidden_size=5, out_hidden_size=7, downsample_ratio=3)
    al = DeepseekV4Aligner(vc)
    n_h, n_w, C = 4, 5, 5
    x = mx.random.normal((n_h * n_w, C))
    got = np.array(al.unfold(x, n_h, n_w))
    g = np.zeros((6, 6, C), dtype=np.float32)
    g[:n_h, :n_w] = np.array(x).reshape(n_h, n_w, C)
    rows = []
    for bh in range(2):
        for bw in range(2):
            blk = g[bh * 3:bh * 3 + 3, bw * 3:bw * 3 + 3]   # (ki, kj, c)
            rows.append(blk.transpose(2, 0, 1).reshape(-1))  # (c, ki, kj)
    assert np.allclose(got, np.stack(rows), atol=1e-6)


def test_vision_rope_tables_shape_and_angles():
    # reference get_vision_cos_sin(n_h, n_w, dim=head_dim//2, theta):
    # [N, dim] = [h * f_0..dim/2-1, w * f_0..dim/2-1], row-major patches
    cos, sin = vision_rope_tables(3, 4, 32, 10000.0)
    assert cos.shape == (12, 32) and sin.shape == (12, 32)
    freqs = 1.0 / (10000.0 ** (np.arange(0, 32, 2) / 32))
    ang = np.concatenate([2 * freqs, 1 * freqs])   # patch (h=2, w=1)
    assert np.allclose(np.array(cos)[2 * 4 + 1], np.cos(ang), atol=1e-5)
    assert np.allclose(np.array(sin)[2 * 4 + 1], np.sin(ang), atol=1e-5)


def test_text_chunks_keep_the_plain_attention_signature():
    """A text-only drafter swaps in an attention without the span kwargs
    (DSpark); text chunks and decode must not pass them. An image chunk on
    such a block is the contract violation and surfaces as TypeError."""
    from gmlx.models.deepseek_v4.dspark import DSparkLocalAttention

    lm, args = _text_model()
    lm.model.layers[0].attn = DSparkLocalAttention(args, 0)
    mx.eval(lm.parameters())
    _randomize_zero_params(lm)
    out = lm(mx.array([[1, 2, 3, 4, 5, 6]]))
    mx.eval(out)
    assert out.shape[:2] == (1, 6) and bool(mx.isfinite(out).all())
    ids, spans = _prompt(args.vocab_size, [3], 1, 3, total=20)
    with pytest.raises(TypeError):
        mx.eval(lm(mx.array([ids]), image_spans=spans))
