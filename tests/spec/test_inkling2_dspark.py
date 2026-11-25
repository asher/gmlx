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

import gmlx.models.inkling2.dspark as ds
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


@pytest.fixture(autouse=True)
def _target_head(monkeypatch):
    # The reference tests read the target's head itself; the tests of the
    # draft head copy set its bits.
    monkeypatch.setattr(ds, "_HEAD_BITS", 0)


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

def _drafter(seed=0, rank=RANK):
    config, _ = _config(meta=_meta(**{"dflash.dspark.markov_rank": rank}),
                        shapes={"markov_w1.weight": (rank, PADDED)})
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


@f32_exact
def test_the_keep_count_is_the_prefix_over_the_floor():
    # A block keeps the leading drafts whose prefix the drafter gives at
    # least row_floor of chance; with no floor it leaves no keep count.
    d = _drafter()
    d.append_context(mx.ones((1, 3, 4 * HIDDEN)))
    toks = d.draft_block(3, None, None, 4, sampler=None, greedy=True)
    assert d.take_row_keep() is None
    hidden = _np(d._draft_hidden(mx.array([[3]])))[0]
    context = _np(d._cache[-1][0]).reshape(-1)
    rows = _reference_chain(d, hidden, 3, context, 3)
    picks = np.array(toks).reshape(-1)
    prefix = np.cumprod([_softmax(r)[t] for r, t in zip(rows, picks)])
    for want in range(4):
        hi = 1.0 if want == 0 else prefix[want - 1]
        lo = prefix[want] if want < 3 else 0.0
        if hi - lo < 1e-3:
            continue
        object.__setattr__(d, "row_floor", float((hi + lo) / 2))
        d.draft_block(3, None, None, 4, sampler=None, greedy=True)
        assert int(d.take_row_keep().item()) == want
        assert d.take_row_keep() is None


def test_unset_the_floor_follows_the_target():
    # With GMLX_INKLING2_ROW_CUT unset, a block leaves a keep count only
    # where the bound target's cut rows read no experts at its width.
    from types import SimpleNamespace

    for saves in (True, False):
        d = _drafter()
        object.__setattr__(d, "_row_target", SimpleNamespace(
            row_cut_saves=lambda rows, saves=saves: saves and rows == 4))
        d.append_context(mx.ones((1, 3, 4 * HIDDEN)))
        d.draft_block(3, None, None, 4, sampler=None, greedy=True)
        assert (d.take_row_keep() is not None) == saves


@f32_exact
def test_draft_head_bits_read_an_affine_copy_of_the_head(monkeypatch):
    # GMLX_INKLING2_DRAFT_HEAD_BITS: the draft rows read an 8-bit copy of
    # the target's head and of the folded correction, close to the head's
    # own rows; the target's head, which the verify reads, is unchanged.
    d = _drafter(rank=32)
    rng = np.random.default_rng(2)
    hidden = mx.array(rng.standard_normal((1, BLOCK, HIDDEN)), dtype=mx.float32)
    d._cache[-1][0] = mx.array(rng.standard_normal((1, 1, HIDDEN)),
                               dtype=mx.float32)
    head = np.array(d.lm_head.weight)
    want, _ = d.chain(hidden, mx.array(9), BLOCK - 1)
    monkeypatch.setattr(ds, "_HEAD_BITS", 8)
    object.__setattr__(d, "_fold", None)
    got, _ = d.chain(hidden, mx.array(9), BLOCK - 1)
    assert d._qhead[3] == 32 and isinstance(d._fold[0], tuple)
    diff = np.abs(_np(got) - _np(want))
    assert 0 < diff.max() < 0.1
    np.testing.assert_array_equal(np.array(d.lm_head.weight), head)


def test_a_rebind_keeps_the_derived_heads_until_the_head_changes(
        monkeypatch):
    # A server binds the drafter once a request: the folded head and the
    # draft head copy are built once and kept while the head's arrays stay.
    from types import SimpleNamespace

    from gmlx.spec.dflash_drafter import DFlashDrafter

    monkeypatch.setattr(ds, "_HEAD_BITS", 8)
    monkeypatch.setattr(DFlashDrafter, "bind", lambda self, t: setattr(
        self, "lm_head", t.lm_head))
    d = _drafter(rank=32)
    target = SimpleNamespace(model=SimpleNamespace(embed_norm=d.embed_norm),
                             lm_head=d.lm_head)
    d._cache[-1][0] = mx.ones((1, 1, HIDDEN))
    for _ in range(2):
        d.bind(target)
        d.chain(mx.ones((1, BLOCK, HIDDEN)), mx.array(3), 2)
        if _ == 0:
            fold, copy = d._fold, d._qhead
    assert d._fold is fold and d._qhead is copy
    target.lm_head = nn.Linear(HIDDEN, PADDED, bias=False)
    d.bind(target)
    assert d._fold is None and d._qhead is None


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


def test_an_accept_commits_the_rings_and_the_context_feature(monkeypatch):
    from gmlx.spec import dflash_drafter as dd

    monkeypatch.setattr(dd, "_COMMIT_ASYNC", True)
    rng = np.random.default_rng(5)
    packed = mx.array(rng.standard_normal((1, 6, 5 * HIDDEN)),
                      dtype=mx.float32)
    whole = _drafter()
    whole.prefill_from_target_hidden(None, packed, None, None, None)
    d = _drafter()
    d.prefill_from_target_hidden(None, packed[:, :4], None, None, None)
    committed = []
    real = mx.async_eval
    monkeypatch.setattr(mx, "async_eval",
                        lambda *a: (committed.append(a[0]), real(*a)))
    # Accepting one draft commits verify rows 0 and 1.
    d.accept_verified_tokens(packed[:, 4:], mx.array([[7]]), 1, [7, 9],
                             sampler=None)
    assert any(x is d._cache[-1][0] for x in committed[-1])
    np.testing.assert_array_equal(_np(d._cache[-1][0]),
                                  _np(whole._cache[-1][0]))
    anchor = mx.array([[5]])
    np.testing.assert_allclose(_np(d._draft_hidden(anchor)),
                               _np(whole._draft_hidden(anchor)),
                               atol=1e-5, rtol=1e-5)


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


def _write_header_only(path, names, meta=None, shapes=None):
    """A drafter GGUF with the released header keys (or ``meta``) and tiny
    tensors, or zeros of their numpy shapes in ``shapes``."""
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "dflash")
    for key, val in (meta or _meta()).items():
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
        w.add_tensor(name, np.zeros((shapes or {}).get(name, (2, 4)),
                                    np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.mark.parametrize("bits", [0, 4])
def test_pricing_counts_what_the_drafter_derives(tmp_path, monkeypatch,
                                                 bits):
    """A loaded drafter holds its file plus what it derives from the
    target's head on its first draft: the folded head with a float32 bias
    row and, at draft head bits, the head's affine copy. Other drafters
    hold their file only."""
    import os

    from gmlx.load.discovery import drafter_resident_bytes

    monkeypatch.setattr(ds, "_HEAD_BITS", bits)
    d = _drafter(rank=32)
    d.set_dtype(mx.bfloat16)
    d._cache[-1][0] = mx.ones((1, 1, HIDDEN), mx.bfloat16)
    d.chain(mx.ones((1, BLOCK, HIDDEN), mx.bfloat16), mx.array(3), 2)
    fold_w, fold_b = d._fold
    assert fold_b.dtype == mx.float32
    built = sum(a.nbytes for a in (
        [fold_b] + (list(fold_w[:3]) if bits else [fold_w])
        + (list(d._qhead[:3]) if bits else [])))

    path = tmp_path / "dspark.gguf"
    _write_header_only(path, ("dspark_hidden_out.weight", "markov_w1.weight"),
                       _meta(**{"dflash.dspark.markov_rank": 32}))
    assert drafter_resident_bytes(str(path)) - os.path.getsize(path) == built
    other = tmp_path / "plain.gguf"
    _write_header_only(other, ("markov_w1.weight",))
    assert drafter_resident_bytes(str(other)) == os.path.getsize(other)


def test_the_full_layer_reads_the_block_rows_in_place(monkeypatch):
    """The full layer writes the block rows into its cache buffer past the
    offset instead of copying the context. Over several rounds the drafts,
    and the context rows the next round writes over them, match the
    copying path."""
    import gmlx.models.inkling2.dspark as D
    from gmlx.cache import kv_append

    if not kv_append.INPLACE:
        pytest.skip("needs mlx_kquant.kv_append")
    rng = np.random.default_rng(4)
    rounds = [rng.standard_normal((1, n, 4 * HIDDEN)).astype(np.float32)
              for n in (5, 1, 3, 2)]

    def run():
        d = _drafter()
        drafts = []
        for ctx in rounds:
            d.append_context(mx.array(ctx))
            drafts.append(d.draft_block(3, None, None, 4, sampler=None,
                                        greedy=True).tolist())
        full = [c for c, kind in zip(d._cache, d.config.layer_types)
                if kind != "sliding_attention"]
        assert full
        return drafts, [np.array(a) for c in full for a in c.state]

    monkeypatch.setattr(kv_append, "INPLACE", False)
    want_drafts, want_kv = run()
    monkeypatch.setattr(kv_append, "INPLACE", True)
    used = []
    real = D._block_in_place
    monkeypatch.setattr(D, "_block_in_place",
                        lambda *a: used.append(real(*a)) or used[-1])
    got_drafts, got_kv = run()
    assert len(used) == len(rounds) and all(r is not None for r in used)
    assert got_drafts == want_drafts
    for g, w in zip(got_kv, want_kv):
        np.testing.assert_array_equal(g, w)


# --- llama.cpp mainline layout -------------------------------------------------

MAINLINE_ROOT = (
    "joint_proj.bias", "joint_proj.weight", "markov_ctx_mix.weight",
    "markov_ctx_norm.weight", "markov_w1.weight", "markov_w2.bias",
    "markov_w2.weight", "mask_embd.weight", "output_norm.weight",
    "token_embd_norm.weight",
)
MAINLINE_BLK = tuple(n for n in BLK_LEAVES if n != "dflare_norm.weight") + (
    "ctx_mix.weight", "ctx_norm.weight")


def _mainline_names(n_layers=3):
    return {n: None for n in MAINLINE_ROOT + tuple(
        f"blk.{i}.{leaf}" for i in range(n_layers) for leaf in MAINLINE_BLK)}


def _mainline_meta():
    # The mainline converter writes no vocabulary or rank keys and stores
    # the window with the anchor's own row.
    meta = _meta(**{"dflash.attention.sliding_window": WINDOW + 1,
                    "dflash.has_markov_context": True})
    for key in ("dflash.vocab_size", "dflash.vocab_size_unpadded",
                "dflash.dspark.markov_rank", "dflash.attention.causal"):
        del meta[key]
    return meta


def _as_mainline(d) -> dict:
    """Drafter ``d``'s weights as llama.cpp's mainline converter writes
    them: softmaxed tap mixes, renamed norms, one mask row per draft slot,
    the unpadded markov table and the correction folded through the head."""
    params = dict(tree_flatten(d.parameters()))
    gguf = {}
    for name in _names():
        (target,), _, _ = remap_dflash_arrays({name: None}, {}, C)
        gguf[name] = params[target]
    out = {}
    for name, a in gguf.items():
        leaf = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if leaf == "dflare_norm.weight":
            out[name.replace("dflare_norm", "ctx_norm")] = a
        elif name not in ("dspark_fuse.weight", "dspark_norm.weight",
                          "mask_embd.weight", "markov_w1.weight") \
                and not name.startswith("dspark_hidden_out"):
            out[name] = a
    mix = mx.softmax(gguf["dspark_fuse.weight"], axis=-1)
    for i in range(3):
        out[f"blk.{i}.ctx_mix.weight"] = mix[i]
    out["markov_ctx_mix.weight"] = mix[3]
    out["markov_ctx_norm.weight"] = gguf["dspark_norm.weight"]
    out["mask_embd.weight"] = mx.broadcast_to(gguf["mask_embd.weight"],
                                              (BLOCK - 1, HIDDEN))
    out["markov_w1.weight"] = gguf["markov_w1.weight"][:VOCAB]
    head = d.lm_head.weight[:VOCAB]
    out["markov_w2.weight"] = head @ gguf["dspark_hidden_out.weight"]
    out["markov_w2.bias"] = head @ gguf["dspark_hidden_out.bias"]
    out["token_embd_norm.weight"] = d.embed_norm.weight
    return out


def _mainline_drafter(d, rank=RANK):
    from gmlx.spec.mtp_load import inkling2_mainline_arrays

    arrays = inkling2_mainline_arrays("d.gguf", _as_mainline(d), 3)
    config, _ = _config(meta=_mainline_meta(), arrays=_mainline_names(),
                        shapes={"markov_w1.weight": (rank, VOCAB)})
    m = Inkling2DSparkDrafter(config)
    weights, _, stats = remap_dflash_arrays(arrays, {}, C)
    m.load_weights(list(weights.items()), strict=True)
    m.embed_tokens, m.lm_head, m.embed_norm = d.embed_tokens, d.lm_head, d.embed_norm
    m._cache = m.make_cache()
    return m


def test_the_mainline_layout_is_its_own_container_too():
    assert dflash_container(_mainline_names()) == C
    # Qwen's DSpark has a markov table and its own head, no tap mixes.
    assert dflash_container({"markov_w1.weight": None, "markov_w2.weight": None,
                             "blk.0.attn_q_norm.weight": None}) == "dflash_dspark"


def test_the_mainline_layout_drafts_what_the_unsloth_layout_drafts():
    """The same weights in llama.cpp's mainline layout: the same window,
    the same block rows, and the same corrected logits and picks."""
    d = _drafter()
    m = _mainline_drafter(d)
    assert m.config.sliding_window == d.config.sliding_window == WINDOW + 1
    rng = np.random.default_rng(5)
    caps = mx.array(rng.standard_normal((1, 9, 4 * HIDDEN)), dtype=mx.float32)
    for x in (d, m):
        x.append_context(caps)
    hd, hm = d._draft_hidden(mx.array([[3]])), m._draft_hidden(mx.array([[3]]))
    np.testing.assert_allclose(_np(hm), _np(hd), rtol=1e-5, atol=1e-5)
    rd, td = d.chain(hd, mx.array(3), BLOCK - 1)
    rm, tm = m.chain(hd, mx.array(3), BLOCK - 1)
    assert rm.shape == rd.shape == (BLOCK - 1, VOCAB)
    np.testing.assert_allclose(_np(rm), _np(rd), rtol=1e-4, atol=1e-4)
    assert tm.tolist() == td.tolist()


def test_mainline_mask_rows_must_be_one_trained_mask():
    from gmlx.spec.mtp_load import inkling2_mainline_arrays

    arrays = _as_mainline(_drafter())
    arrays["mask_embd.weight"] = mx.arange(
        (BLOCK - 1) * HIDDEN, dtype=mx.float32).reshape(BLOCK - 1, HIDDEN)
    with pytest.raises(ValueError, match="mask"):
        inkling2_mainline_arrays("d.gguf", arrays, 3)


@pytest.mark.parametrize("bits", [0, 4])
def test_pricing_counts_what_a_mainline_drafter_derives(tmp_path, monkeypatch,
                                                        bits):
    """The mainline file carries the folded head, so the drafter derives
    nothing at bits 0 and only the affine copies at draft head bits."""
    import os

    from gmlx.load.discovery import drafter_resident_bytes

    monkeypatch.setattr(ds, "_HEAD_BITS", bits)
    m = _mainline_drafter(_drafter(rank=32), rank=32)
    m.set_dtype(mx.bfloat16)
    m.markov_w2.bias = m.markov_w2.bias.astype(mx.float32)  # fp32_keep
    m._cache[-1][0] = mx.ones((1, 1, HIDDEN), mx.bfloat16)
    m.chain(mx.ones((1, BLOCK, HIDDEN), mx.bfloat16), mx.array(3), 2)
    fold_w, fold_b = m._fold
    assert fold_b is m.markov_w2.bias
    built = (sum(a.nbytes for a in list(fold_w[:3]) + list(m._qhead[:3]))
             if bits else 0)

    meta = _mainline_meta()
    path = tmp_path / "dspark.gguf"
    _write_header_only(path, tuple(_mainline_names()), meta,
                       shapes={"markov_w1.weight": (VOCAB, 32),
                               "markov_w2.weight": (VOCAB, 32)})
    assert drafter_resident_bytes(str(path)) - os.path.getsize(path) == built
