"""DSpark heads on the DFlash backbone (the Ternary Bonsai drafters): the
container split, the closed remap with the flat confidence row and the
biases, the header keys, the log-SNR features, the bigram chain against a
numpy transcription of llama.cpp's ``build_dspark_markov_head``, the
confidence cut, and the engine contract on a tiny random qwen3_5 target."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import gmlx.spec.mtp_load as mtp_load
from gmlx.spec.dflash_drafter import (
    DFlashDrafter,
    DSparkDFlashDrafter,
    _LOG_SNR_FEATURES,
    log_snr_features,
)
from gmlx.spec.mtp_load import (
    _dflash_config_from_meta,
    dflash_container,
    remap_dflash_arrays,
)

from test_dflash2_drafter import (
    BLOCK,
    CAPTURE,
    N_GEN,
    _engine_reference,
    _engine_walk,
    _target,
    _tcfg,
)
from test_vlm_mtp_gating import _cfg, _top

HIDDEN, VOCAB, RANK = 64, 128, 8
MASK = 7
BLK_LEAVES = (
    "attn_k.weight", "attn_k_norm.weight", "attn_norm.weight",
    "attn_output.weight", "attn_q.weight", "attn_q_norm.weight",
    "attn_v.weight", "ffn_down.weight", "ffn_gate.weight", "ffn_norm.weight",
    "ffn_up.weight",
)
ROOT_LEAVES = (
    "enc.output_norm.weight", "fc.weight", "output_norm.weight",
    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight",
    "conf_proj.bias",
)
SNR_LEAVES = ("log_snr_fc1.weight", "log_snr_fc1.bias", "log_snr_fc2.weight",
              "log_snr_fc2.bias")
KQUANT = {"attn_k", "attn_output", "attn_q", "attn_v", "ffn_down", "ffn_gate",
          "ffn_up", "fc", "markov_w1", "markov_w2", "log_snr_fc1",
          "log_snr_fc2"}


def _meta(**over):
    meta = {
        "general.architecture": "dflash",
        "dflash.block_count": 2,
        "dflash.embedding_length": HIDDEN,
        "dflash.feed_forward_length": 64,
        "dflash.attention.head_count": 4,
        "dflash.attention.head_count_kv": 2,
        "dflash.attention.key_length": 16,
        "dflash.attention.layer_norm_rms_epsilon": 1e-6,
        "dflash.rope.freq_base": 10000.0,
        "dflash.context_length": 4096,
        "dflash.block_size": 4,
        "dflash.markov_rank": RANK,
        "dflash.confidence_head": True,
        "dflash.confidence_head_with_markov": True,
        "dflash.target_layers": [1, 3],
        "tokenizer.ggml.mask_token_id": MASK,
    }
    meta.update(over)
    return meta


def _snr_meta(**over):
    return _meta(**{"dflash.log_snr_conditioning": True,
                    "dflash.min_log_snr": -9.0, "dflash.max_log_snr": 9.0,
                    **over})


def _target_dict(**over):
    d = {"model_type": "qwen3_5", "num_hidden_layers": 4, "hidden_size": HIDDEN,
         "vocab_size": VOCAB, "max_position_embeddings": 2048}
    d.update(over)
    return d


def _skeleton(n_layers=2, *, snr=False, kquant=True):
    arrays, kq = {}, {}
    leaves = ROOT_LEAVES + (SNR_LEAVES if snr else ()) + tuple(
        f"blk.{i}.{leaf}" for i in range(n_layers) for leaf in BLK_LEAVES)
    for name in leaves:
        arrays[name] = None
        stem = name.split(".")[-2] if name.endswith((".weight", ".bias")) else name
        if kquant and name.endswith(".weight") and stem.split(".")[-1] in KQUANT:
            kq[name] = "q8_0"
            arrays[name[:-len(".weight")] + ".scales"] = None
    return arrays, kq


# --- container and remap -----------------------------------------------------

def test_markov_on_plain_layers_is_the_dflash_backbone():
    arrays, _ = _skeleton()
    assert dflash_container(arrays) == "dflash_dspark"
    assert dflash_container({**arrays, "blk.0.attn_q_a.weight": None}) == "dspark"
    assert dflash_container({**arrays, "output_hc_fn.weight": None}) == "dspark"
    plain = {n: None for n in arrays if not n.startswith(("markov", "conf"))}
    assert dflash_container(plain) == "muse_glimmer"


@pytest.mark.parametrize("snr", [False, True])
def test_remap_covers_every_param_exactly(snr):
    arrays, kq = _skeleton(snr=snr)
    weights, meta, stats = remap_dflash_arrays(arrays, kq, "dflash_dspark")
    config, _ = _dflash_config_from_meta(
        "d.gguf", _snr_meta() if snr else _meta(), _target_dict(),
        "dflash_dspark", arrays=arrays)
    params = {k for k, _ in tree_flatten(DSparkDFlashDrafter(config).parameters())}
    mapped = {k for k in weights if not k.endswith(".scales")}
    assert mapped == params
    assert stats["mapped"] == len(mapped)
    assert "conf_proj.bias" in mapped and "markov_w2.weight" in meta
    assert ("log_snr_fc1.bias" in mapped) is snr


def test_remap_reshapes_the_flat_confidence_row_and_is_closed():
    arrays, kq = _skeleton()
    arrays["conf_proj.weight"] = mx.zeros((HIDDEN + RANK,))
    weights, _, _ = remap_dflash_arrays(arrays, kq, "dflash_dspark")
    assert weights["conf_proj.weight"].shape == (1, HIDDEN + RANK)
    with pytest.raises(RuntimeError, match="unknown tensor"):
        remap_dflash_arrays({**arrays, "selector_hidden.weight": None}, kq,
                            "dflash_dspark")


# --- config ------------------------------------------------------------------

def test_config_reads_the_dspark_keys_and_adds_the_anchor_row():
    arrays, _ = _skeleton()
    config, ids = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(),
                                           "dflash_dspark", arrays=arrays)
    assert ids == (0, 2)
    assert (config.markov_rank, config.confidence_head) == (RANK, True)
    assert config.log_snr_range is None and config.sample_from_anchor
    # A 4-row block drafts 4 tokens; the engine's block total counts the
    # verify row on top.
    assert (config.block_size, config.native_block_size) == (5, 5)
    assert config.is_dspark and not config.is_dflash2
    no_anchor, _ = _dflash_config_from_meta(
        "d.gguf", _meta(**{"dflash.sample_from_anchor": False}), _target_dict(),
        "dflash_dspark", arrays=arrays)
    assert (no_anchor.block_size, no_anchor.sample_from_anchor) == (4, False)


def test_config_takes_the_rank_from_the_table_and_checks_the_heads():
    arrays, _ = _skeleton(snr=True)
    arrays["markov_w1.weight"] = mx.zeros((VOCAB, RANK))
    config, _ = _dflash_config_from_meta(
        "d.gguf", _snr_meta(**{"dflash.markov_rank": None}), _target_dict(),
        "dflash_dspark", arrays=arrays)
    assert config.markov_rank == RANK and config.log_snr_range == (-9.0, 9.0)
    with pytest.raises(ValueError, match="conf_proj.weight is missing"):
        _dflash_config_from_meta(
            "d.gguf", _meta(), _target_dict(), "dflash_dspark",
            arrays={n: v for n, v in arrays.items() if "conf_proj" not in n})
    with pytest.raises(ValueError, match="min_log_snr < max_log_snr"):
        _dflash_config_from_meta(
            "d.gguf", _snr_meta(**{"dflash.max_log_snr": -9.0}), _target_dict(),
            "dflash_dspark", arrays=arrays)
    with pytest.raises(ValueError, match="log_snr_fc2.weight missing"):
        _dflash_config_from_meta(
            "d.gguf", _snr_meta(), _target_dict(), "dflash_dspark",
            arrays={n: v for n, v in arrays.items() if "fc2" not in n})


# --- drafter -----------------------------------------------------------------

def _config(cfg, *, n_layers=2, block_size=BLOCK + 1, confidence=True,
            snr=False, anchor=True):
    from gmlx.spec.dflash_drafter import DFlashConfig

    return DFlashConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        block_size=block_size,
        native_block_size=block_size,
        mask_token_id=MASK,
        target_layer_ids=list(CAPTURE),
        num_target_layers=cfg.num_hidden_layers,
        layer_types=["sliding_attention"] * n_layers,
        sliding_window=512,
        markov_rank=RANK,
        confidence_head=confidence,
        log_snr_range=(-9.0, 9.0) if snr else None,
        sample_from_anchor=anchor,
    )


def _drafter(lm=None, **kw):
    drafter = DSparkDFlashDrafter(_config(_tcfg(), **kw))
    mx.eval(drafter.parameters())
    if lm is not None:
        drafter.reset(lm)
    return drafter


def test_drafter_needs_a_markov_head():
    from gmlx.spec.dflash_drafter import DFlashConfig

    cfg = dataclasses.replace(_config(_tcfg()), markov_rank=0)
    assert isinstance(cfg, DFlashConfig)
    with pytest.raises(ValueError, match="markov_rank > 0"):
        DSparkDFlashDrafter(cfg)


def test_log_snr_features_match_the_reference_loop():
    """llama.cpp fills ``[n_tokens, 128]``: level 1000 on the anchor, 0 on
    the masks, sin over the first half and cos over the second."""
    rows = 4
    got = np.array(log_snr_features(rows))
    half = _LOG_SNR_FEATURES // 2
    want = np.zeros((rows, _LOG_SNR_FEATURES), dtype=np.float32)
    for pos in range(rows):
        tt = 1000.0 if pos == 0 else 0.0
        for i in range(half):
            freq = np.exp(-np.log(10000.0) * i / half)
            want[pos, i] = np.sin(tt * freq)
            want[pos, half + i] = np.cos(tt * freq)
    np.testing.assert_allclose(got, want, atol=2e-3, rtol=0)


def test_log_snr_embedding_is_added_once_per_row_and_memoized():
    drafter = _drafter(snr=True)
    toks = mx.array([[3, MASK, MASK, MASK]])
    base = DFlashDrafter._embed_input_tokens
    drafter.embed_tokens = lambda t: mx.zeros(t.shape + (HIDDEN,))
    plain = base(drafter, toks)
    got = drafter._embed_input_tokens(toks)
    feat = log_snr_features(4)
    want = drafter.log_snr_fc2(mx.maximum(drafter.log_snr_fc1(feat), 0) * 0 +
                               drafter.log_snr_fc1(feat) *
                               mx.sigmoid(drafter.log_snr_fc1(feat)))
    mx.eval(got, want, plain)
    assert mx.abs(got[0] - plain[0] - want).max().item() < 1e-5
    assert 4 in drafter._snr_embed
    # Row 0 differs from the masks, and the masks share one embedding.
    assert mx.abs(got[0, 0] - got[0, 1]).max().item() > 1e-4
    assert mx.abs(got[0, 1] - got[0, 2]).max().item() < 1e-6


def _chain_reference(w1, w2, conf_w, conf_b, hidden, logits, anchor, mask):
    """numpy transcription of build_dspark_markov_head for one block."""
    rows, confs = [], []
    prev = int(anchor)
    for i in range(logits.shape[0]):
        m = w1[prev]
        row = logits[i].astype(np.float64) + w2 @ m
        row[mask] = -np.inf
        rows.append(row)
        feat = np.concatenate([hidden[i], m])
        confs.append(1.0 / (1.0 + np.exp(-(conf_w @ feat + conf_b))))
        prev = int(np.argmax(row))
    return np.stack(rows), np.array(confs)


def test_chain_matches_the_reference_and_excludes_the_mask(monkeypatch):
    drafter = _drafter()
    monkeypatch.setattr(drafter, "_confidence_tau", 0.5)
    mx.random.seed(3)
    hidden = mx.random.normal((4, HIDDEN))
    logits = mx.random.normal((4, VOCAB)) * 3
    logits = logits.at[:, MASK].add(100.0)   # the mask must never win
    rows, confs = drafter.chain(hidden, logits, mx.array(5))
    mx.eval(rows, confs)
    w1 = np.array(drafter.markov_w1.weight, dtype=np.float64)
    w2 = np.array(drafter.markov_w2.weight, dtype=np.float64)
    cw = np.array(drafter.conf_proj.weight, dtype=np.float64)[0]
    cb = float(drafter.conf_proj.bias[0].item())
    want_rows, want_confs = _chain_reference(
        w1, w2, cw, cb, np.array(hidden, dtype=np.float64),
        np.array(logits, dtype=np.float64), 5, MASK)
    got = np.array(rows, dtype=np.float64)
    assert np.all(np.isneginf(got[:, MASK])) and rows.dtype == mx.float32
    finite = np.arange(VOCAB) != MASK
    np.testing.assert_allclose(got[:, finite], want_rows[:, finite], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(confs), want_confs, atol=1e-4, rtol=0)
    assert np.argmax(got, axis=-1).tolist() == np.argmax(want_rows, axis=-1).tolist()
    # The chain depends on the picks: a different anchor moves row 1 too.
    rows2, _ = drafter.chain(hidden, logits, mx.array(6))
    delta = np.abs(np.array(rows2)[:, finite] - got[:, finite])
    assert delta.max() > 1e-3


def test_confidence_off_skips_the_head():
    drafter = _drafter()
    assert drafter._confidence_tau == 0.0
    rows, confs = drafter.chain(mx.zeros((3, HIDDEN)), mx.zeros((3, VOCAB)), mx.array(1))
    assert confs is None and rows.shape == (3, VOCAB)


def test_draft_block_rows_and_the_anchor_row_drafts():
    lm = _target()
    drafter = _drafter(lm)
    block = drafter._block_tokens(3, BLOCK + 1, mx.int32)
    assert block.tolist() == [[3] + [MASK] * (BLOCK - 1)]
    drafts = drafter.draft_block(3, None, None, BLOCK + 1, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, BLOCK)
    assert MASK not in drafts[0].tolist()
    with pytest.raises(RuntimeError, match="drafts at most"):
        drafter.draft_block(3, None, None, BLOCK + 2, None, greedy=True)
    no_anchor = _drafter(lm, anchor=False, block_size=BLOCK)
    got = no_anchor.draft_block(3, None, None, BLOCK, None, greedy=True)
    mx.eval(got)
    assert got.shape == (1, BLOCK - 1)


def test_confidence_cut_keeps_the_confident_prefix(monkeypatch):
    lm = _target()
    drafter = _drafter(lm)
    seen = {}
    orig = drafter.chain

    def chain(hidden, logits, anchor):
        rows, _ = orig(hidden, logits, anchor)
        seen["rows"] = rows
        return rows, mx.array([0.9, 0.8, 0.2, 0.95])

    monkeypatch.setattr(drafter, "chain", chain)
    monkeypatch.setattr(drafter, "_confidence_tau", 0.5)
    drafts = drafter.draft_block(3, None, None, BLOCK + 1, None, greedy=True)
    mx.eval(drafts)
    assert drafts[0].tolist() == mx.argmax(seen["rows"][:2], axis=-1).tolist()
    monkeypatch.setattr(drafter, "chain",
                        lambda h, lg, a: (orig(h, lg, a)[0], mx.array([0.1, 0.9, 0.9, 0.9])))
    drafts = drafter.draft_block(3, None, None, BLOCK + 1, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, 1)


def test_engine_rounds_emit_the_greedy_chain():
    lm = _target()
    prompt = mx.array([[1, 2, 3, 4, 5]])
    ref = _engine_reference(lm, prompt, N_GEN)
    assert len(set(ref)) >= 4
    got = _engine_walk(lm, _drafter(lm), prompt, N_GEN)
    assert got[:N_GEN] == ref


# --- load-through on the tiny pair -------------------------------------------

def _random_weights(config):
    drafter = DSparkDFlashDrafter(config)
    mx.random.seed(1)
    return {k: mx.random.normal(v.shape) * 0.05
            for k, v in tree_flatten(drafter.parameters())}


def _wire(arrays, weights):
    named, _, _ = remap_dflash_arrays({n: n for n in arrays}, {}, "dflash_dspark")
    wire = {named[k]: v for k, v in weights.items()}
    wire["conf_proj.weight"] = wire["conf_proj.weight"].reshape(-1)
    return wire


@pytest.mark.parametrize("snr", [False, True])
def test_loader_builds_binds_and_arms_the_target(tmp_path, snr):
    import gmlx.models.qwen35.owned as qwen35_owned

    cfg = dataclasses.replace(_cfg(), tie_word_embeddings=False)
    mx.random.seed(2)
    lm = qwen35_owned.language_model_class("qwen3_5")(cfg, _top())
    mx.eval(lm.parameters())
    target_dict = _target_dict(num_hidden_layers=cfg.num_hidden_layers)
    meta = _snr_meta() if snr else _meta()
    arrays, _ = _skeleton(snr=snr)
    config, _ = _dflash_config_from_meta("d.gguf", meta, target_dict,
                                         "dflash_dspark", arrays=arrays)
    random = _random_weights(config)
    wire = _wire(arrays, random)
    gguf = tmp_path / "d.gguf"
    gguf.write_bytes(b"")
    logs = []
    drafter = mtp_load._load_dflash_dspark_drafter(
        str(gguf), lm, target_dict, arrays=wire, kquant_meta={}, meta=meta,
        log=lambda m, *a, **k: logs.append(str(m)))
    assert isinstance(drafter, DSparkDFlashDrafter)
    assert lm._dflash_capture == (0, 2)
    assert drafter.lm_head is lm.lm_head
    assert (drafter.mtp_width_cap, drafter.mtp_width_limit) == (1, 1)
    got = drafter.conf_proj.weight
    mx.eval(got)
    assert got.shape == (1, HIDDEN + RANK)
    assert mx.abs(got.astype(mx.float32) - random["conf_proj.weight"]).max().item() < 1e-2
    assert any("dflash_dspark layers=2 targets=(0, 2)" in m for m in logs)
    # Full-attention layers, as the Bonsai drafters ship: the ring fills
    # from the armed target's packed hidden before the first block.
    drafter.reset(lm)
    prompt = mx.array([[1, 2, 3, 4, 5]])
    hid, _, _ = lm.speculative_verify_hidden(prompt, lm.make_cache())
    drafter.prefill_from_target_hidden(prompt, hid, 3, None, greedy=True)
    drafts = drafter.draft_block(3, None, None, 5, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, 4)


def test_companion_routing_takes_the_container(monkeypatch, tmp_path):
    arrays, kq = _skeleton()
    calls = {}
    monkeypatch.setattr(mtp_load, "load_gguf_wire_bytes",
                        lambda p, zero_copy=True: (arrays, kq, "dflash", _meta(), {}))
    monkeypatch.setattr(mtp_load, "_load_dflash_dspark_drafter",
                        lambda *a, **k: calls.setdefault("dspark", (a, k)))
    mtp_load._load_dflash_drafter(str(tmp_path / "d.gguf"), SimpleNamespace(),
                                  _target_dict(), log=lambda *a, **k: None)
    assert "dspark" in calls
