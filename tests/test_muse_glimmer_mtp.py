"""Muse Glimmer speculative target: the packed-hidden capture seam and the
verify/rollback walk, on a tiny random model (no GGUF, no weights).

The seam widens every engine-facing hidden to ``[trunk | cap ...]`` so the
drafter can read the target's residuals without an engine change. Two things
have to hold for that to be safe, and both are load-bearing:

- the logits hooks must slice the trunk back out, and
- the slice must be materialized before it reaches the logit head, which is a
  quantized kernel on a real model and reads the buffer directly. A lazy
  strided view hands it the packed strides and it reads the wrong rows -
  the target emitted token soup and draft acceptance fell to ~3%. Nothing in
  a float model reproduces that, so the invariant is pinned directly here and
  the numeric end of it rides the integration tier.
"""

import mlx.core as mx
import pytest

from gmlx.config_synth import synthesize_config
from gmlx.muse_glimmer_model import ModelArgs, ensure_registered
from gmlx.muse_glimmer_mtp import MuseGlimmerSpecLM

from test_config_synth import _MUSE_GLIMMER_SHAPES, _muse_glimmer_meta

CAPTURE = (0, 2)
N_GEN = 16
BLOCK = 4
# Weight init is unseeded, and about a fifth of draws leave the tiny model's
# top-2 logits tied within the floor below at step 0, which makes the identity
# claim vacuous. Pin the draw: this one holds a 1.6e-2 minimum top-2 margin
# across all N_GEN reference steps.
SEED = 25
# The verify path derives its tokens through a block SDPA (qL = drafts + 1)
# whose rounding differs from the 1-token decode path by at most 3e-7 in logit
# space. A gap above this floor is a real divergence, not rounding.
GREEDY_TIE_TOL = 1e-3


def _build():
    mx.random.seed(SEED)
    ensure_registered()
    cfg = synthesize_config(_muse_glimmer_meta(),
                            tensor_shapes=_MUSE_GLIMMER_SHAPES)
    lm = MuseGlimmerSpecLM(ModelArgs.from_dict(cfg))
    mx.eval(lm.parameters())
    return lm, cfg


def _packed_width(cfg):
    return cfg["hidden_size"] * (1 + len(CAPTURE))


def test_capture_arms_the_packed_hidden():
    lm, cfg = _build()
    ids = mx.array([[1, 2, 3, 4]])
    plain, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    assert plain.shape[-1] == cfg["hidden_size"]

    lm.set_dflash_capture(CAPTURE)
    packed, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    assert packed.shape[-1] == _packed_width(cfg)
    assert lm._dflash_capture == CAPTURE


def test_trunk_slice_recovers_the_unpacked_hidden():
    lm, cfg = _build()
    ids = mx.array([[1, 2, 3, 4]])
    bare, _ = lm.speculative_verify_hidden(ids, lm.make_cache())
    lm.set_dflash_capture(CAPTURE)
    packed, _ = lm.speculative_verify_hidden(ids, lm.make_cache())

    trunk = lm._dflash_trunk(packed)
    mx.eval(bare, trunk)
    assert trunk.shape == bare.shape
    assert float(mx.abs(trunk - bare).max().item()) == 0.0
    # and the hooks agree with the unpacked logits
    lm.set_dflash_capture(())
    ref = lm.speculative_logits_from_hidden(bare)
    lm.set_dflash_capture(CAPTURE)
    got = lm.speculative_logits_from_hidden(packed)
    mx.eval(ref, got)
    assert float(mx.abs(ref - got).max().item()) == 0.0


def test_trunk_is_materialized_before_the_logit_head(monkeypatch):
    """White-box on purpose: a float head cannot show the difference, but the
    real head is a quantized kernel that reads the buffer directly."""
    lm, cfg = _build()
    lm.set_dflash_capture(CAPTURE)
    calls = []
    real = mx.contiguous
    monkeypatch.setattr(
        mx, "contiguous", lambda x, *a, **k: (calls.append(x.shape), real(x, *a, **k))[1])
    packed = mx.zeros((1, 3, _packed_width(cfg)))
    lm.speculative_logits_from_hidden(packed)
    assert calls, (
        "the packed trunk slice must be materialized before the logit head; "
        "a lazy strided view makes the quantized kernel read the wrong rows"
    )


def test_argmax_hook_matches_the_logits_hook():
    lm, _ = _build()
    lm.set_dflash_capture(CAPTURE)
    packed, _ = lm.speculative_verify_hidden(
        mx.array([[1, 2, 3, 4]]), lm.make_cache())
    logits = lm.speculative_logits_from_hidden(packed)
    am = lm.speculative_argmax_from_hidden(packed)
    mx.eval(logits, am)
    assert am.tolist() == mx.argmax(logits, axis=-1).tolist()


def test_rollback_trims_every_layer_cache():
    lm, _ = _build()
    cache = lm.make_cache()
    lm.speculative_verify_hidden(mx.array([[1, 2, 3, 4, 5, 6]]), cache)
    before = [c.offset for c in cache]
    lm.speculative_verify_hidden(mx.array([[7] * BLOCK]), cache)
    assert all(c.offset == b + BLOCK for c, b in zip(cache, before))
    # accepted=1 keeps the bonus row plus one draft; the rest is rejected
    lm.rollback_speculative_cache(cache, None, 1, BLOCK)
    assert all(c.offset == b + 2 for c, b in zip(cache, before))


def test_rollback_is_a_noop_when_the_whole_block_is_accepted():
    lm, _ = _build()
    cache = lm.make_cache()
    lm.speculative_verify_hidden(mx.array([[1, 2, 3, 4]]), cache)
    lm.speculative_verify_hidden(mx.array([[5] * BLOCK]), cache)
    offsets = [c.offset for c in cache]
    lm.rollback_speculative_cache(cache, None, BLOCK - 1, BLOCK)
    assert [c.offset for c in cache] == offsets


@pytest.mark.parametrize("armed", [False, True])
def test_verify_walk_is_token_identical_to_greedy(armed):
    """The engine contract: whatever the drafter proposes, the walk emits the
    target's own greedy tokens. Driven with deliberately wrong drafts so every
    round takes the reject-and-rollback path.

    Parametrized over the capture seam because the bug this guards only
    appeared with capture armed - the packed slice reached the head unevaluated.
    """
    lm, cfg = _build()
    vocab = cfg["vocab_size"]
    prompt = mx.array([[1, 2, 3, 4, 5]])

    cache = lm.make_cache()
    h = lm.model(prompt, cache)
    ref_logits = lm._spec_logits(h)[0, -1]
    ref, margins = [], []
    for _ in range(N_GEN):
        top = mx.sort(ref_logits)[-2:]
        margins.append(float((top[1] - top[0]).item()))
        t = int(mx.argmax(ref_logits).item())
        ref.append(t)
        ref_logits = lm._spec_logits(lm.model(mx.array([[t]]), cache))[0, -1]

    assert min(margins) > GREEDY_TIE_TOL, (
        "the pinned draw no longer has an unambiguous greedy chain; choose "
        "another SEED rather than weakening the identity claim"
    )

    if armed:
        lm.set_dflash_capture(CAPTURE)
    cache2 = lm.make_cache()
    hid, _ = lm.speculative_verify_hidden(prompt, cache2)
    tok = int(lm.speculative_argmax_from_hidden(hid)[0, -1].item())
    got, accepts = [tok], []
    while len(got) < N_GEN:
        # One past the target's own pick, so position 0 always rejects. Ids stay
        # in vocab: an out-of-range id is an out-of-bounds gather that reads
        # uninitialized memory on the CPU backend, and a NaN landing in a
        # rejected slot does not stay there - it propagates through the masked
        # SDPA into the accepted row, whose argmax then collapses to 0.
        drafts = [(ref[len(got)] + 1 + i) % vocab for i in range(BLOCK - 1)]
        hid, _ = lm.speculative_verify_hidden(
            mx.array([[tok] + drafts]), cache2)
        rows = lm.speculative_argmax_from_hidden(hid)[0].tolist()
        accepted = 0
        for i, d in enumerate(drafts):
            if int(rows[i]) != d:
                break
            accepted += 1
        accepts.append(accepted)
        got.extend(drafts[:accepted])
        tok = int(rows[accepted])
        got.append(tok)
        lm.rollback_speculative_cache(cache2, None, accepted, BLOCK)

    assert got == ref
    assert accepts == [0] * len(accepts), (
        "every round was meant to reject at position 0 and roll back")


# --- the drafter side of the same seam ---------------------------------------

def _build_drafter(cfg, n_layers=2, block_size=BLOCK, native_block_size=None):
    from gmlx.muse_glimmer_dflash import (
        MuseGlimmerDFlashConfig,
        MuseGlimmerDFlashDrafter,
    )

    return MuseGlimmerDFlashDrafter(MuseGlimmerDFlashConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=1024,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        block_size=block_size,
        native_block_size=native_block_size,
        mask_token_id=7,
        target_layer_ids=list(CAPTURE),
        num_target_layers=cfg["num_hidden_layers"],
        layer_types=["sliding_attention"] * n_layers,
        sliding_window=512,
        draft_window_size=512,
        final_logit_softcapping=cfg["final_logit_softcapping"],
        output_multiplier=cfg["output_multiplier"],
    ))


def test_drafter_captures_are_materialized_for_the_quantized_fc(monkeypatch):
    """Same trap as the logit head: ``fc`` is a quantized 5*hidden -> hidden
    matmul, so the trailing slice of the packed hidden must not reach it lazily."""
    lm, cfg = _build()
    drafter = _build_drafter(cfg)
    calls = []
    real = mx.contiguous
    monkeypatch.setattr(
        mx, "contiguous", lambda x, *a, **k: (calls.append(x.shape), real(x, *a, **k))[1])
    drafter._captures(mx.zeros((1, 3, _packed_width(cfg))))
    assert calls, "the packed capture slice must be materialized before fc"


def test_drafter_captures_reject_an_unpacked_hidden():
    """A target whose ``_dflash_capture`` was never armed hands over a bare
    trunk; that must fail loudly rather than matmul against garbage."""
    lm, cfg = _build()
    drafter = _build_drafter(cfg)
    with pytest.raises(ValueError, match="packed hidden width"):
        drafter._captures(mx.zeros((1, 3, cfg["hidden_size"])))


def test_drafter_captures_take_the_trailing_block():
    lm, cfg = _build()
    drafter = _build_drafter(cfg)
    h = cfg["hidden_size"]
    packed = mx.concatenate(
        [mx.zeros((1, 2, h)), mx.ones((1, 2, h)), mx.full((1, 2, h), 2.0)],
        axis=-1)
    caps = drafter._captures(packed)
    mx.eval(caps)
    assert caps.shape[-1] == h * len(CAPTURE)
    assert float(caps[0, 0, 0].item()) == 1.0        # first capture
    assert float(caps[0, 0, h].item()) == 2.0        # second, in order


def test_drafter_satisfies_the_protocol():
    from gmlx.drafter_protocol import validate_drafter

    lm, cfg = _build()
    drafter = _build_drafter(cfg)
    mx.eval(drafter.parameters())
    drafter.bind(lm)
    validate_drafter(drafter)
    assert drafter.uses_shared_kv is False
    assert drafter.requires_owned_engine is True


def test_a_shallow_default_still_drafts_the_trained_block_on_request():
    """The loader defaults the runtime depth below the checkpoint's block. A
    deeper ``--draft-block-size`` must reach the drafter and produce that many
    drafts, which is what the depth ceiling exists for."""
    from gmlx.spec_helpers import _resolve_block_total

    lm, cfg = _build()
    drafter = _build_drafter(cfg, block_size=2, native_block_size=BLOCK)
    mx.eval(drafter.parameters())
    drafter.reset(lm)

    assert _resolve_block_total(drafter, None) == 2
    block_total = _resolve_block_total(drafter, BLOCK)
    assert block_total == BLOCK

    drafts = drafter.draft_block(3, None, None, block_total, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, BLOCK - 1)


def _load_drafter_config(cfg, monkeypatch, block_size):
    """Drive the GGUF loader far enough to capture the config it builds."""
    from gmlx import muse_glimmer_dflash as mgd
    from gmlx import mtp_load

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(config):
        captured["config"] = config
        raise _Stop

    monkeypatch.setattr(mgd, "MuseGlimmerDFlashDrafter", _capture)
    meta = {
        "dflash.target_layers": [1, 3],
        "dflash.block_size": block_size,
        "tokenizer.ggml.mask_token_id": 7,
        "dflash.feed_forward_length": 64,
        "dflash.attention.head_count": 4,
        "dflash.attention.head_count_kv": 2,
        "dflash.attention.key_length": 16,
        "dflash.attention.layer_norm_rms_epsilon": 1e-6,
        "dflash.rope.freq_base": 10000.0,
        "dflash.attention.sliding_window": 512,
    }
    with pytest.raises(_Stop):
        mtp_load._load_muse_glimmer_dflash_drafter(
            "draft.gguf", None, cfg,
            arrays={"blk.0.attn_q.weight": None, "blk.1.attn_q.weight": None},
            kquant_meta={}, meta=meta, log=lambda *a, **k: None)
    return captured["config"]


def test_the_loader_stamps_the_gguf_block_as_the_drafter_ceiling(monkeypatch):
    """The runtime depth is a tuned default; the checkpoint's trained block is
    the ceiling a deeper ``--draft-block-size`` is allowed to reach."""
    from gmlx.mtp_load import _MUSE_GLIMMER_DFLASH_BLOCK_DEFAULT as TUNED

    _, cfg = _build()
    deep = _load_drafter_config(cfg, monkeypatch, TUNED + 16)
    assert deep.native_block_size == TUNED + 16
    assert deep.block_size == TUNED

    shallow = _load_drafter_config(cfg, monkeypatch, 3)
    assert shallow.native_block_size == 3
    assert shallow.block_size == 3


def test_an_undeclared_ceiling_keeps_the_configured_depth():
    from gmlx.spec_helpers import _resolve_block_total

    _, cfg = _build()
    drafter = _build_drafter(cfg, block_size=BLOCK)
    assert drafter._native_block_size == BLOCK
    assert _resolve_block_total(drafter, BLOCK + 8) == BLOCK
