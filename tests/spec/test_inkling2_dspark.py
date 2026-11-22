"""The Inkling 2 DSpark drafter: the container split, the closed remap,
the header reading (taps as given, the anchor row's window, the unpadded
vocabulary), the recurrent head against a numpy transcription of
llama.cpp's ``build_inkling2_head``, the context feature and the sliding
window of the draft block."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from f32_exact import f32_exact  # noqa: E402

from gmlx.models.inkling2.dspark import Inkling2DSparkDrafter, first_argmax
from gmlx.spec.mtp_load import (
    _dflash_config_from_meta,
    dflash_container,
    remap_dflash_arrays,
)

C = "dflash_dspark_inkling2"
HIDDEN, RANK, WINDOW, BLOCK = 32, 6, 4, 8
VOCAB, PADDED = 64, 70
VISITS = 6
BLK_LEAVES = (
    "attn_k.weight", "attn_k_ctx.weight", "attn_k_ctx_norm.weight",
    "attn_k_norm.weight", "attn_norm.weight", "attn_output.weight",
    "attn_q.weight", "attn_q_norm.weight", "attn_v.weight",
    "attn_v_ctx.weight", "dflare_norm.weight", "ffn_down.weight",
    "ffn_gate.weight", "ffn_norm.weight", "ffn_up.weight",
)
ROOT_LEAVES = (
    "dspark_fuse.weight", "dspark_hidden_out.bias", "dspark_hidden_out.weight",
    "dspark_norm.weight", "joint_proj.bias", "joint_proj.weight",
    "markov_w1.weight", "mask_embd.weight", "output_norm.weight",
)
SHAPES = {"markov_w1.weight": (RANK, PADDED)}


def _meta(**over):
    # The header keys of the released drafter, at a tiny size.
    meta = {
        "general.architecture": "dflash",
        "dflash.block_count": 3,
        "dflash.context_length": 4096,
        "dflash.embedding_length": HIDDEN,
        "dflash.feed_forward_length": 48,
        "dflash.attention.head_count": 4,
        "dflash.attention.head_count_kv": 2,
        "dflash.attention.key_length": 8,
        "dflash.attention.value_length": 8,
        "dflash.rope.freq_base": 1000000.0,
        "dflash.attention.layer_norm_rms_epsilon": 1e-6,
        "dflash.vocab_size": PADDED,
        "dflash.vocab_size_unpadded": VOCAB,
        "dflash.block_size": BLOCK,
        "dflash.target_layers": [1, 3, 5, VISITS],
        "dflash.dspark.markov_rank": RANK,
        "dflash.sample_from_anchor": False,
        "dflash.has_confidence_head": False,
        "dflash.attention.causal": False,
        "dflash.attention.sliding_window": WINDOW,
        "dflash.attention.sliding_window_pattern": [True, False, True],
        "dflash.rope.dimension_count": 8,
        "dflash.final_logit_softcapping": 30.0,
        "tokenizer.ggml.mask_token_id": 0,
    }
    meta.update(over)
    return meta


def _target(**over):
    d = {"model_type": "inkling2", "hidden_size": HIDDEN,
         "num_hidden_layers": 4, "vocab_size": PADDED,
         "unpadded_vocab_size": VOCAB, "max_position_embeddings": 4096,
         "layer_visits": [[i, "full"] for i in range(VISITS)],
         "final_logit_softcapping": 30.0}
    d.update(over)
    return d


def _names(n_layers=3):
    return {n: None for n in ROOT_LEAVES + tuple(
        f"blk.{i}.{leaf}" for i in range(n_layers) for leaf in BLK_LEAVES)}


def _config(meta=None, target=None, arrays=None, shapes=SHAPES):
    return _dflash_config_from_meta(
        "d.gguf", meta or _meta(), target or _target(), C,
        arrays=_names() if arrays is None else arrays, shapes=shapes)


# --- container, remap, header ------------------------------------------------

def test_the_hidden_out_head_is_its_own_container():
    names = _names()
    assert dflash_container(names) == C
    del names["dspark_hidden_out.weight"], names["dspark_hidden_out.bias"]
    assert dflash_container(names) == "dflash_dspark"


def test_remap_covers_every_param_exactly_and_is_closed():
    arrays = _names()
    weights, _, stats = remap_dflash_arrays(arrays, {}, C)
    config, _ = _config()
    params = {k for k, _ in tree_flatten(Inkling2DSparkDrafter(config).parameters())}
    assert set(weights) == params and stats["mapped"] == len(params)
    with pytest.raises(RuntimeError, match="unknown tensor"):
        remap_dflash_arrays({**arrays, "conf_proj.weight": None}, {}, C)


def test_config_takes_the_taps_as_given_and_widens_the_window():
    config, ids = _config()
    # Visit inputs, and the visit count for the normed final hidden state:
    # no minus-one shift.
    assert ids == (1, 3, 5, VISITS)
    # The anchor row sees WINDOW context rows and itself.
    assert config.sliding_window == WINDOW + 1
    assert (config.vocab_size, config.padded_vocab_size) == (VOCAB, PADDED)
    assert (config.block_size, config.native_block_size) == (4, BLOCK)
    assert config.markov_rank == RANK and not config.sample_from_anchor
    assert config.layer_types == ["sliding_attention", "full_attention",
                                  "sliding_attention"]


def test_the_tap_bound_is_the_visit_count_here_and_unchanged_elsewhere():
    with pytest.raises(ValueError, match="strictly increasing"):
        _config(_meta(**{"dflash.target_layers": [1, 3, VISITS + 1]}))
    from test_dflash_dspark import _meta as qmeta
    from test_dflash_dspark import _skeleton, _target_dict

    arrays, _ = _skeleton()
    # A plain target's last tap is its layer count (the input of no layer).
    _dflash_config_from_meta("d.gguf", qmeta(**{"dflash.target_layers": [1, 4]}),
                             _target_dict(), "dflash_dspark", arrays=arrays)
    with pytest.raises(ValueError, match="strictly increasing"):
        _dflash_config_from_meta(
            "d.gguf", qmeta(**{"dflash.target_layers": [1, 5]}),
            _target_dict(), "dflash_dspark", arrays=arrays)


@pytest.mark.parametrize("over", [
    {"dflash.attention.causal": True},
    {"dflash.sample_from_anchor": True},
    {"dflash.attention.sliding_window": 0},
    {"dflash.block_size": 4},
    {"dflash.final_logit_softcapping": 0.0},
    {"dflash.target_layers": [1, 3, 5]},
])
def test_config_refuses_a_block_the_drafter_was_not_trained_with(over):
    with pytest.raises(ValueError, match="Inkling 2 DSpark"):
        _config(_meta(**over))


def test_config_refuses_a_markov_table_off_the_target_head():
    with pytest.raises(ValueError, match="markov_w1 has 64 rows"):
        _config(shapes={"markov_w1.weight": (RANK, VOCAB)})


def test_config_refuses_another_family():
    with pytest.raises(ValueError, match="not qwen3_5"):
        _config(target=_target(model_type="qwen3_5"))


# --- the drafter -------------------------------------------------------------

def _drafter(seed=0):
    config, _ = _config()
    d = Inkling2DSparkDrafter(config)
    rng = np.random.default_rng(seed)
    d.update(_random_like(d.parameters(), rng))
    # The parts bind takes from the target.
    d.embed_tokens = nn.Embedding(PADDED, HIDDEN)
    d.lm_head = nn.Linear(HIDDEN, PADDED, bias=False)
    d.embed_norm = nn.RMSNorm(HIDDEN)
    d.embed_tokens.weight = mx.array(rng.standard_normal((PADDED, HIDDEN)),
                                     dtype=mx.float32)
    d.lm_head.weight = mx.array(rng.standard_normal((PADDED, HIDDEN)) * 0.5,
                                dtype=mx.float32)
    d._cache = d.make_cache()
    return d


def _random_like(tree, rng):
    if isinstance(tree, dict):
        return {k: _random_like(v, rng) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_random_like(v, rng) for v in tree]
    return mx.array(rng.standard_normal(tree.shape) * 0.3, dtype=mx.float32)


def _np(a):
    return np.array(a.astype(mx.float32))


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _rms(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _reference_chain(d, hidden, anchor, context, n, picks=None):
    """build_inkling2_head, row by row, with the first-maximum pick over
    the unpadded vocabulary (dspark.cpp graph_inkling2). ``picks`` forces
    the token each row feeds the next."""
    out = _np(d.lm_head.weight)
    w1 = _np(d.markov_w1.weight)
    jw, jb = _np(d.joint_proj.weight), _np(d.joint_proj.bias)
    hw, hb = _np(d.hidden_out.weight), _np(d.hidden_out.bias)
    state = np.zeros(RANK, np.float32)
    prev, rows = anchor, []
    for i in range(1, n + 1):
        joint = np.concatenate([state, w1[prev], hidden[i], context]) @ jw.T + jb
        gate = 1 / (1 + np.exp(-joint[:RANK]))
        cand = np.tanh(joint[RANK:2 * RANK])
        state = cand + gate * (state - cand)
        correction = np.tanh(joint[2 * RANK:]) @ hw.T + hb
        logits = hidden[i] @ out.T + correction @ out.T
        logits = (np.tanh(logits / 30.0) * 30.0)[:VOCAB]
        rows.append(logits)
        prev = (int(np.argmax(logits)) if picks is None  # first maximum
                else picks[i - 1])
    return np.stack(rows)


@f32_exact
def test_the_recurrent_head_matches_the_reference():
    d = _drafter()
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((1, BLOCK, HIDDEN)).astype(np.float32)
    context = rng.standard_normal(HIDDEN).astype(np.float32)
    d._cache[-1][0] = mx.array(context[None, None])
    rows, toks = d.chain(mx.array(hidden), mx.array(9), BLOCK - 1)
    got = _np(rows)
    want = _reference_chain(d, hidden[0], 9, context, BLOCK - 1)
    assert got.shape == (BLOCK - 1, VOCAB)
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
    # Each row's pick feeds the next row.
    assert np.array_equal(np.array(toks), want.argmax(-1))


@f32_exact
def test_a_sampled_draft_is_the_token_the_next_row_reads():
    d = _drafter()
    d.append_context(mx.ones((1, 3, 4 * HIDDEN)))
    drawn, seen = [], []

    def sampler(logits):
        # The lowest-scored token: never the first maximum.
        seen.append(_np(logits)[0])
        tok = mx.argmin(logits, axis=-1)
        drawn.append(int(tok.reshape(-1)[0]))
        return tok

    toks = d.draft_block(3, None, None, 4, sampler=sampler, greedy=False)
    assert toks.reshape(-1).tolist() == drawn and len(drawn) == 3
    hidden = _np(d._draft_hidden(mx.array([[3]])))[0]
    context = _np(d._cache[-1][0]).reshape(-1)
    # Row i + 1 is conditioned on the drawn token of row i.
    want = _reference_chain(d, hidden, 3, context, 3, picks=drawn)
    np.testing.assert_allclose(np.stack(seen), want, atol=2e-4, rtol=2e-4)


def test_drafts_never_pick_a_padded_row():
    d = _drafter()
    # The padded rows of the head would win every row.
    w = np.array(d.lm_head.weight)
    w[VOCAB:] = 40.0
    d.lm_head.weight = mx.array(w)
    hidden = mx.ones((1, BLOCK, HIDDEN))
    d._cache[-1][0] = mx.ones((1, 1, HIDDEN))
    rows, _ = d.chain(hidden, mx.array(3), BLOCK - 1)
    assert rows.shape[-1] == VOCAB
    toks = d.draft_block(3, None, None, 4, sampler=None, greedy=True)
    assert toks.shape == (1, 3) and int(mx.max(toks)) < VOCAB


def test_the_block_rows_commit_once_and_leave_the_drafts_unchanged(
        monkeypatch):
    def drafts():
        d = _drafter()
        d.append_context(mx.ones((1, 3, 4 * HIDDEN)))
        return d.draft_block(3, None, None, 4, sampler=None,
                             greedy=True).tolist()

    real = mx.async_eval
    monkeypatch.setattr(mx, "async_eval", lambda *a: None)
    want = drafts()
    committed = []
    monkeypatch.setattr(mx, "async_eval",
                        lambda *a: (committed.append(a), real(*a)))
    assert drafts() == want
    assert len(committed) == 1


def test_first_argmax_takes_the_first_of_a_tie():
    assert first_argmax(mx.array([[1.0, 3.0, 3.0, 2.0]])).tolist() == [1]
    assert first_argmax(mx.array([5.0, 1.0, 5.0])).item() == 0


def test_the_context_feature_is_the_last_row_through_the_final_fuse_row():
    d = _drafter()
    rng = np.random.default_rng(2)
    taps = rng.standard_normal((1, 5, 4 * HIDDEN)).astype(np.float32)
    d.append_context(mx.array(taps))
    w = _softmax(_np(d.dspark_fuse)[-1])
    mixed = (taps[0, -1].reshape(4, HIDDEN) * w[:, None]).sum(0)
    want = _rms(mixed, _np(d.dspark_norm.weight))
    np.testing.assert_allclose(_np(d._cache[-1][0]).reshape(-1), want,
                               atol=1e-5, rtol=1e-5)


def test_sliding_layers_see_the_anchor_rows_window(monkeypatch):
    d = _drafter()
    d.append_context(mx.ones((1, 20, 4 * HIDDEN)))
    seen = []
    sdpa = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append(int(k.shape[2]))
        assert kw.get("mask") is None  # every block row sees every key
        return sdpa(q, k, v, **kw)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    d._draft_hidden(mx.array([[3]]))
    # Sliding layers: the anchor row's WINDOW context rows for every row,
    # plus the block. The full layer: all 20 context rows plus the block.
    assert seen == [WINDOW + BLOCK, 20 + BLOCK, WINDOW + BLOCK]


def test_chunked_seeding_matches_one_shot_and_counts_every_row():
    from gmlx.spec.mtp_drafter import _cache_offset

    # A full layer reads the whole prompt, so nothing caps the capture.
    assert _drafter().hidden_capture_limit is None
    rng = np.random.default_rng(4)
    rows = 13
    packed = mx.array(rng.standard_normal((1, rows, 5 * HIDDEN)),
                      dtype=mx.float32)
    one = _drafter()
    one.prefill_from_target_hidden(None, packed, None, None, None)
    seeded = _drafter()
    seed = seeded.make_cache()
    seeded.seed_chunk(None, packed[:, :6], seed)
    seeded.seed_chunk(None, packed[:, 6:10], seed)
    # The engine adopts a seed whose offset is the streamed length.
    assert _cache_offset(seed) == 10
    seeded.restore_kv(seed)
    seeded.prefill_from_target_hidden(None, packed[:, 10:], None, None, None)
    anchor = mx.array([[5]])
    np.testing.assert_allclose(_np(seeded._draft_hidden(anchor)),
                               _np(one._draft_hidden(anchor)),
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(_np(seeded._cache[-1][0]),
                                  _np(one._cache[-1][0]))


def _write_header_only(path, names):
    """A drafter GGUF with the released header keys and tiny tensors."""
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "dflash")
    for key, val in _meta().items():
        if key == "general.architecture":
            continue
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, int):
            w.add_uint32(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_array(key, val)
    for name in names:
        w.add_tensor(name, np.zeros((2, 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_pricing_counts_the_folded_head(tmp_path):
    """A loaded drafter holds its file plus the folded head it builds on
    the first draft: [markov_rank, padded vocab] in bf16 and a float32
    bias row. Other drafters hold their file only."""
    import os

    from gmlx.load.discovery import drafter_resident_bytes

    d = _drafter()
    fold_w, fold_b = d._folded_head()
    assert fold_w.shape == (RANK, PADDED) and fold_b.shape == (PADDED,)
    assert fold_b.dtype == mx.float32

    path = tmp_path / "dspark.gguf"
    _write_header_only(path, ("dspark_hidden_out.weight", "markov_w1.weight"))
    assert (drafter_resident_bytes(str(path)) - os.path.getsize(path)
            == RANK * PADDED * 2 + PADDED * 4)
    other = tmp_path / "plain.gguf"
    _write_header_only(other, ("markov_w1.weight",))
    assert drafter_resident_bytes(str(other)) == os.path.getsize(other)
