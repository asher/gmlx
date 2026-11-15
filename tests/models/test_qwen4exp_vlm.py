"""qwen4exp VLM wrapper: config synth, model build, mrope parity."""

import mlx.core as mx
import mlx.nn as nn

import gmlx.load.vlm as gvlm
from gmlx.load.config_synth import synthesize_config
from gmlx.models.qwen4_exp.model import (
    Model as TextModel,
    ModelArgs,
    ensure_registered,
)
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

_MM_META = {
    "clip.projector_type": "qwen3vl_merger",
    "clip.vision.patch_size": 16,
    "clip.vision.image_size": 768,
    "clip.vision.block_count": 2,
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 64,
    "clip.vision.projection_dim": 64,
    "clip.vision.attention.head_count": 2,
    "clip.vision.spatial_merge_size": 2,
}


def test_resolve_and_config_synth():
    assert gvlm.resolve_vlm_model_type("qwen4exp", _MM_META) == "qwen4_exp"
    cfg = gvlm.synthesize_vlm_config(
        "qwen4_exp", _qwen4exp_meta(True, True), _QWEN4EXP_SHAPES, _MM_META)
    assert cfg["model_type"] == "qwen4_exp"
    assert cfg["vision_config"]["spatial_merge_size"] == 2
    assert cfg["vision_config"]["deepstack_visual_indexes"] == []


def _tiny_vlm():
    ensure_registered()
    import gmlx.models.qwen4_exp.vlm_model as vm

    vm.ensure_registered()
    text = dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES))
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "out_hidden_size": text["hidden_size"], "num_heads": 2,
            "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "num_position_embeddings": 4,
            "in_channels": 3, "image_size": 64,
            "deepstack_visual_indexes": [],
        },
        "vocab_size": text["vocab_size"],
        "image_token_id": 20, "video_token_id": 21,
        "vision_start_token_id": 18, "vision_end_token_id": 19,
    }
    model, _ = gvlm.build_vlm_model(cfg)
    mx.random.seed(0)
    flat = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        flat[k] = (mx.random.normal(v.shape) * 0.05).astype(v.dtype)
    model.load_weights(list(flat.items()), strict=False)
    mx.eval(model.parameters())
    return model


def test_build_and_text_forward_matches_text_model():
    """Text-only through the VLM wrapper == the vendored text Model."""
    model = _tiny_vlm()
    text_args = ModelArgs.from_dict(
        dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)))
    ref = TextModel(text_args)
    weights = dict(nn.utils.tree_flatten(model.language_model.parameters()))
    ref.load_weights([(k, v) for k, v in weights.items()], strict=True)

    ids = mx.random.randint(0, text_args.vocab_size, (1, 10))
    cache = model.make_cache()
    out = model.language_model(ids, cache=cache).logits
    ref_out = ref(ids, cache=ref.make_cache())
    assert mx.abs(out - ref_out).max() < 1e-5


def test_mrope_positions_equal_streams_match_text_path():
    """position_ids with equal t/h/w streams reproduce the offset path
    through prefill + decode (cache-position block rope included)."""
    model = _tiny_vlm()
    lm = model.language_model
    vocab = lm.args.vocab_size
    ids = mx.random.randint(0, vocab, (1, 11))
    nxt = mx.random.randint(0, vocab, (1, 1))

    c_text = lm.make_cache()
    a = lm(ids, cache=c_text).logits
    a2 = lm(nxt, cache=c_text).logits

    c_pos = lm.make_cache()
    pos = mx.broadcast_to(mx.arange(11, dtype=mx.int32)[None, None], (3, 1, 11))
    b = lm(ids, cache=c_pos, position_ids=pos).logits
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    b2 = lm(nxt, cache=c_pos).logits
    assert mx.abs(a - b).max() < 2e-4
    assert mx.abs(a2 - b2).max() < 2e-4
    lm._rope_deltas = None


def test_get_rope_index_borrowed_semantics():
    model = _tiny_vlm()
    lm = model.language_model
    # 4 image patches -> merge 2 -> 1 soft token at position 2.
    ids = mx.array([[1, 18, 20, 19, 5, 6]])
    grid = mx.array([[1, 2, 2]])
    pos, deltas = lm.get_rope_index(ids, image_grid_thw=grid)
    assert pos.shape == (3, 1, 6)
    assert deltas.shape == (1, 1)
    # streams differ inside the image, equal outside
    p = pos[:, 0].tolist()
    assert p[0][:2] == p[1][:2] == p[2][:2]


def test_mlx_vlm_loader_resolves_qwen4_exp_to_the_gmlx_module():
    import gmlx.models.qwen4_exp.vlm_model as vm
    from gmlx._vendor.mlx_vlm.utils import get_model_and_args

    assert get_model_and_args({"model_type": "qwen4_exp"})[0] is vm


def test_language_model_carries_the_mtp_spec_hooks():
    # mtp_load's VLM branch probes these on model.language_model and refuses
    # text-only MTP if any is missing; the load then fails outright.
    lm = _tiny_vlm().language_model
    for hook in ("speculative_logits_from_hidden",
                 "speculative_argmax_from_hidden",
                 "speculative_verify_hidden", "rollback_speculative_cache",
                 "chunked_prefill_policy"):
        assert callable(getattr(lm, hook)), hook


def test_spec_hooks_shared_with_text_speclm():
    """One mixin, both carriers: the vlm LanguageModel and the text SpecLM
    must run the same hook code, not parallel copies."""
    from gmlx.models.qwen4_exp.mtp import Qwen4ExpSpecHooks, Qwen4ExpSpecLM
    import gmlx.models.qwen4_exp.vlm_model as vm

    assert issubclass(Qwen4ExpSpecLM, Qwen4ExpSpecHooks)
    assert issubclass(vm.LanguageModel, Qwen4ExpSpecHooks)
    for hook in ("speculative_logits_from_hidden",
                 "speculative_argmax_from_hidden",
                 "speculative_verify_hidden", "rollback_speculative_cache",
                 "chunked_prefill_policy"):
        assert getattr(Qwen4ExpSpecLM, hook) is getattr(vm.LanguageModel, hook)


def test_return_hidden_yields_the_four_stream_hidden():
    """return_hidden hands back the pre-mixer [B,S,4,D] streams whose
    speculative_logits_from_hidden reproduces the plain logits."""
    model = _tiny_vlm()
    lm = model.language_model
    ids = mx.random.randint(0, lm.args.vocab_size, (1, 6))
    out = lm(ids, cache=lm.make_cache())
    spec = lm(ids, cache=lm.make_cache(), return_hidden=True)
    assert float(mx.abs(spec.logits - out.logits).max()) < 1e-5
    (streams,) = spec.hidden_states
    assert streams.ndim == 4 and streams.shape[2] == lm.args.hc_count
    relogits = lm.speculative_logits_from_hidden(streams)
    assert float(mx.abs(relogits - out.logits).max()) < 1e-5


def test_skip_logits_leaves_out_the_head():
    model = _tiny_vlm()
    lm = model.language_model
    ids = mx.random.randint(0, lm.args.vocab_size, (1, 6))
    out = lm(ids, cache=lm.make_cache())
    spec = lm(ids, cache=lm.make_cache(), return_hidden=True, skip_logits=True)
    assert spec.logits is None
    assert lm(ids, cache=lm.make_cache(), skip_logits=True).logits is None
    relogits = lm.speculative_logits_from_hidden(spec.hidden_states[-1])
    assert float(mx.abs(relogits - out.logits).max()) < 1e-5


def test_verify_positions_follow_the_resolved_mrope():
    """After an image turn set _rope_deltas, the verify forward must run at
    the same positions plain decode does (spec vs plain logits equal)."""
    model = _tiny_vlm()
    lm = model.language_model
    vocab = lm.args.vocab_size
    ids = mx.random.randint(0, vocab, (1, 9))
    nxt = mx.random.randint(0, vocab, (1, 2))

    lm._rope_deltas = mx.array([[3]], dtype=mx.int32)
    c_plain = lm.make_cache()
    lm(ids, cache=c_plain)
    plain = lm(nxt, cache=c_plain).logits

    c_spec = lm.make_cache()
    lm(ids, cache=c_spec)
    streams, _, sink = lm.speculative_verify_hidden(nxt, c_spec)
    spec = lm.speculative_logits_from_hidden(streams)
    lm._rope_deltas = None
    assert float(mx.abs(spec - plain).max()) < 2e-4
    assert sink  # rollback sink recorded


def test_qsa_cache_pos_tracks_ik_across_midstep_growth():
    """pos/ik lockstep when the buffers grow at a mid-step prev.

    Verify rollback trims land the offset mid-step; the next growth event
    truncates ik to prev before extending, and pos must be sized against
    the SAME truncated base or it ends up short of ik and later writes
    clamp ([broadcast_shapes] (3,1,n) vs (3,1,<n) on serve turn 3)."""
    from gmlx.models.qwen4_exp.model import QSAKVCache

    c = QSAKVCache(ratio=4)
    step = c.step
    D, IDX = 4, 8

    def push(n, start):
        k = mx.zeros((1, 1, n, D))
        v = mx.zeros((1, 1, n, D))
        ik = mx.zeros((1, n, IDX))
        pos = mx.broadcast_to(
            (start + mx.arange(n, dtype=mx.int32))[None, None], (3, 1, n))
        c.update_and_fetch_qsa(k, v, ik, pos=pos)

    push(step, 0)              # fills exactly one step block
    push(4, step)              # grows at prev % step == 0
    assert c.pos.shape[2] == c.ik.shape[1]
    c.trim(2)                  # rollback lands mid-step
    prev = c.offset
    assert prev % step != 0
    push(step, prev)           # growth at mid-step prev: truncation path
    assert c.pos.shape[2] == c.ik.shape[1]
    # The rows just written carry the pushed positions.
    got = c.pos[0, 0, prev:prev + step]
    want = prev + mx.arange(step, dtype=mx.int32)
    assert bool(mx.all(got == want).item())


def _peaked_qsa(lm):
    """Norms of 3 on the QSA layer's q and k make its attention peaked, so
    the context a sparse row leaves out, and a wrong position, show in the
    logits."""
    for layer in lm.layers:
        if not layer.is_linear and layer.self_attn.ratio > 0:
            attn = layer.self_attn
            attn.q_norm.weight = mx.full(attn.q_norm.weight.shape, 3.0)
            attn.k_norm.weight = mx.full(attn.k_norm.weight.shape, 3.0)


def test_batched_decode_slices_mrope_positions_per_row():
    """Two rows decoding together past the indexer budget, with explicit
    mrope positions: the QSA batch cache gives each row its own slice of
    the position block, and each row answers as it does alone."""
    from gmlx._vendor.mlx_lm.generate import _merge_caches

    model = _tiny_vlm()
    lm = model.language_model
    _peaked_qsa(lm)
    vocab = lm.args.vocab_size
    mx.random.seed(5)
    lens = (15, 13)
    prompts = [mx.random.randint(0, vocab, (1, n)) for n in lens]
    nxt = mx.random.randint(0, vocab, (2, 1))

    def pos(starts, n):
        s = mx.array(starts, dtype=mx.int32)[:, None] + mx.arange(n)[None]
        return mx.broadcast_to(s[None].astype(mx.int32), (3, len(starts), n))

    def prefilled(i):
        c = lm.make_cache()
        lm(prompts[i], cache=c, position_ids=pos([0], lens[i]))
        return c

    alone = [lm(nxt[i:i + 1], cache=prefilled(i),
                position_ids=pos([lens[i]], 1)).logits for i in range(2)]
    batch = _merge_caches([prefilled(0), prefilled(1)])
    got = lm(nxt, cache=batch, position_ids=pos(list(lens), 1)).logits
    for i in range(2):
        scale = float(mx.abs(alone[i]).max())
        # the batched projections round unlike a one-row GEMV
        assert float(mx.abs(got[i:i + 1] - alone[i]).max()) < 2e-3 * scale


def test_batched_decode_after_image_turns_takes_each_rows_delta():
    """Two rows after image turns decode together with their own mrope
    deltas, one per row as the decode batch carries them: each row's
    positions are its own offset plus its own delta, and each row answers
    as it does alone."""
    from gmlx._vendor.mlx_lm.generate import _merge_caches

    model = _tiny_vlm()
    lm = model.language_model
    _peaked_qsa(lm)
    vocab = lm.args.vocab_size
    mx.random.seed(9)
    lens, deltas = (15, 13), (3, -2)
    prompts = [mx.random.randint(0, vocab, (1, n)) for n in lens]
    nxt = mx.random.randint(0, vocab, (2, 1))

    def prefilled(i):
        c = lm.make_cache()
        lm(prompts[i], cache=c)
        return c

    def delta(*ds):
        return mx.array([[d] for d in ds], dtype=mx.int32)

    alone = [lm(nxt[i:i + 1], cache=prefilled(i),
                rope_deltas=delta(deltas[i])).logits for i in range(2)]
    batch = _merge_caches([prefilled(0), prefilled(1)])
    got = lm(nxt, cache=batch, rope_deltas=delta(*deltas)).logits
    for i in range(2):
        scale = float(mx.abs(alone[i]).max())
        # the batched projections round unlike a one-row GEMV
        assert float(mx.abs(got[i:i + 1] - alone[i]).max()) < 2e-3 * scale
