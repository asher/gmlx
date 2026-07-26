"""Contract pins for the mlx-vlm text-forward hook surface (qwen3_5, gemma4).

Step 1 of the forward-ownership plan: record the hook behavior an owned
forward must reproduce, as executable checks against tiny random models
rather than against reading. Every pin is an internal-consistency claim
(hook A equals a composition of hook B and the forward; a rollback replay
matches a sequential decode), so the pins hold both against stock mlx-vlm
and against numerics-preserving gmlx seam patches that may already be
installed by earlier test files in the same process.

Per-arch surface pinned here (the plan's hook table):
  qwen3_5: output holder, skip_logits, capture_layer_ids/hidden_sink/
           gdn_sink, speculative_verify_logits / speculative_verify_hidden /
           speculative_logits_from_hidden / speculative_argmax_from_hidden,
           rollback_speculative_cache (KV trim + GDN state restore),
           chunked_prefill_policy, _rope_deltas text zeroing.
  gemma4:  output holder (pre-final-norm hidden capture), skip_final_norm,
           capture_layer_ids, caller-supplied sinks, shared_kv_sink,
           speculative_draft_hidden / speculative_logits_from_hidden,
           rollback_speculative_cache, chunked_prefill_policy.

ds4/hy3 SpecLMs implement only the four middle-row hooks; those are already
pinned end-to-end by test_deepseek_v4_mtp.py / test_hy_v3_mtp.py greedy
identity, and are out of scope here.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")
pytest.importorskip("mlx_vlm.models.gemma4.language")

from mlx_vlm.models.gemma4.config import TextConfig as G4TextConfig
from mlx_vlm.models.gemma4.language import LanguageModel as G4LanguageModel
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel

ATOL = 2e-3  # f32 path-divergence floor (block SDPA / chunked GDN vs step)

# The qwen3_5 GDN forward dispatches Metal-only kernels; the gemma4 tests
# and the policy-table checks run on any device.
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")
# Replay pins compare states built through a qL=3 forward against states
# built through qL=1 steps: the KV entries themselves differ by kernel-path
# rounding, so the bound is looser. Real rollback bugs (wrong trim, stale
# GDN state) miss by orders of magnitude, not by rounding.
REPLAY_ATOL = 1e-2


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# tiny models
# ---------------------------------------------------------------------------


def _q35_lm():
    mx.random.seed(11)
    cfg = Q35TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )
    # get_rope_index dereferences the vision config and the multimodal token
    # ids on every fresh text forward (mrope grid math) -- a text-only
    # construction still needs these stubs. Pinned here as behavior an owned
    # forward must either reproduce or explicitly drop.
    top = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )
    lm = Q35LanguageModel(cfg, top)
    mx.eval(lm.parameters())
    return lm


def _g4_lm():
    mx.random.seed(13)
    cfg = G4TextConfig(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=6,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        num_key_value_heads=2,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        sliding_window=32,
        sliding_window_pattern=3,
        tie_word_embeddings=True,
    )
    lm = G4LanguageModel(cfg)
    mx.eval(lm.parameters())
    return lm


def _ids(*toks):
    return mx.array([list(toks)])


PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


def _close(a, b, atol=ATOL):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() < atol


# ---------------------------------------------------------------------------
# qwen3_5
# ---------------------------------------------------------------------------


@_NEEDS_GPU
def test_q35_output_holder_and_hidden_head_consistency():
    lm = _q35_lm()
    ids = _ids(*PROMPT)
    out = lm(ids)
    assert out.logits.shape == (1, len(PROMPT), lm.args.vocab_size)
    assert out.hidden_states is None
    assert out.shared_kv_states is None

    out2 = lm(ids, return_hidden=True, return_shared_kv=True)
    assert isinstance(out2.hidden_states, list) and len(out2.hidden_states) == 1
    h = out2.hidden_states[-1]
    assert h.shape == (1, len(PROMPT), lm.args.hidden_size)
    assert out2.shared_kv_states == {}
    # the captured hidden IS the head input: re-projecting it reproduces logits
    assert _close(lm.speculative_logits_from_hidden(h), out2.logits)


@_NEEDS_GPU
def test_q35_skip_logits_still_captures_hidden():
    lm = _q35_lm()
    out = lm(_ids(*PROMPT), return_hidden=True, skip_logits=True)
    assert out.logits is None
    assert out.hidden_states is not None and len(out.hidden_states) == 1


@_NEEDS_GPU
def test_q35_capture_layer_ids_and_gdn_sink():
    lm = _q35_lm()
    n_gdn = sum(1 for lyr in lm.layers if lyr.is_linear)
    assert n_gdn == 3  # interval-4 hybrid: 3 GDN + 1 full-attention
    out = lm(_ids(*PROMPT), capture_layer_ids=[0, 2])
    assert len(out.hidden_states) == 2
    for h in out.hidden_states:
        assert h.shape == (1, len(PROMPT), lm.args.hidden_size)
    # capture mode runs the target-verify path: one gdn_sink entry per GDN layer
    assert out.gdn_states is not None and len(out.gdn_states) == n_gdn


@_NEEDS_GPU
def test_q35_verify_hooks_agree_with_forward():
    lm = _q35_lm()
    prompt = _ids(*PROMPT)
    block = _ids(5, 9, 11)

    caches = []
    for _ in range(3):
        c = lm.make_cache()
        mx.eval(lm(prompt, cache=c).logits)
        caches.append(c)

    ref = lm(block, cache=caches[0], capture_layer_ids=[],
             return_hidden=True, return_shared_kv=True)

    hidden, shared_kv, gdn, sampled = lm.speculative_verify_logits(
        block, caches[1], _greedy)
    assert _close(hidden, ref.hidden_states[-1])
    assert (sampled == _greedy(ref.logits)).all().item()
    assert shared_kv == ref.shared_kv_states == {}
    assert gdn is not None and len(gdn) == len(ref.gdn_states)

    hidden2, _, _ = lm.speculative_verify_hidden(block, caches[2])
    assert _close(hidden2, ref.hidden_states[-1])

    # dense-head hooks compose: argmax_from_hidden == argmax(logits_from_hidden)
    am = lm.speculative_argmax_from_hidden(hidden)
    assert (am == _greedy(lm.speculative_logits_from_hidden(hidden))).all().item()


@_NEEDS_GPU
def test_q35_rollback_replays_to_sequential_state():
    lm = _q35_lm()
    prompt = _ids(*PROMPT)
    t1, t2, g2, g3 = 5, 6, 7, 9

    seq_cache = lm.make_cache()
    mx.eval(lm(prompt, cache=seq_cache).logits)
    seq_after_t1 = lm(_ids(t1), cache=seq_cache).logits
    seq_after_t2 = lm(_ids(t2), cache=seq_cache).logits

    ver_cache = lm.make_cache()
    mx.eval(lm(prompt, cache=ver_cache).logits)
    ver = lm(_ids(t1, g2, g3), cache=ver_cache, capture_layer_ids=[],
             return_hidden=True, return_shared_kv=True)
    # block position 0 sees the same state as the sequential t1 step
    assert _close(ver.logits[:, :1], seq_after_t1, atol=REPLAY_ATOL)

    lm.rollback_speculative_cache(ver_cache, ver.gdn_states, 0, 3)
    ver_after_t2 = lm(_ids(t2), cache=ver_cache).logits
    assert _close(ver_after_t2, seq_after_t2, atol=REPLAY_ATOL)


def test_q35_chunked_prefill_policy_table():
    lm = _q35_lm()
    draft = object()
    assert lm.chunked_prefill_policy(draft_model=None) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp", prefill_kwargs={}) is False
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp",
        prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
    ) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="dflash",
        prefill_kwargs={"capture_layer_ids": [1]}) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="dflash", prefill_kwargs={}) is False
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind=None, prefill_kwargs={}) is True


@_NEEDS_GPU
def test_q35_text_rope_deltas_zeroed():
    lm = _q35_lm()
    ids = mx.array([[3, 17, 42], [9, 8, 7]])
    mx.eval(lm(ids).logits)
    rd = lm._rope_deltas
    assert rd is not None and rd.shape == (2, 1)
    assert mx.abs(rd).max().item() == 0


# ---------------------------------------------------------------------------
# gemma4
# ---------------------------------------------------------------------------


def test_g4_output_holder_prenorm_hidden_and_head():
    lm = _g4_lm()
    ids = _ids(*PROMPT)
    out = lm(ids, return_hidden=True)
    assert len(out.hidden_states) == 1
    h = out.hidden_states[-1]
    # captured hidden is PRE-final-norm: the speculative head hook (which
    # norms internally) reproduces the logits from it
    assert _close(lm.speculative_logits_from_hidden(h), out.logits)
    # draft-hidden hook is exactly the final norm
    assert _close(lm.speculative_draft_hidden(h), lm.model.norm(h), atol=1e-6)


def test_g4_skip_final_norm_matches_prenorm_capture():
    lm = _g4_lm()
    ids = _ids(*PROMPT)
    h_skip = lm.model(ids, skip_final_norm=True)
    out = lm(ids, return_hidden=True)
    assert _close(h_skip, out.hidden_states[-1], atol=1e-6)
    assert _close(lm.model.norm(h_skip), lm.model(ids), atol=1e-6)


def test_g4_capture_layer_ids_and_caller_sink():
    lm = _g4_lm()
    ids = _ids(*PROMPT)
    out = lm(ids, capture_layer_ids=[0, 3])
    # explicit capture set: exactly those layers, no final-layer append
    assert len(out.hidden_states) == 2
    my_sink = []
    out2 = lm(ids, hidden_sink=my_sink, return_hidden=True)
    assert out2.hidden_states is my_sink and len(my_sink) == 1


def test_g4_shared_kv_sink_keys_and_shapes():
    lm = _g4_lm()
    cfg = lm.config
    S = len(PROMPT)
    out = lm(_ids(*PROMPT), return_shared_kv=True)
    kv = out.shared_kv_states
    # [s,s,f,s,s,f] with 2 shared tail layers: one shared consumer per type
    assert set(kv.keys()) == {"sliding_attention", "full_attention"}
    k_slide, v_slide = kv["sliding_attention"][:2]
    assert k_slide.shape == (1, cfg.num_key_value_heads, S, cfg.head_dim)
    k_full, v_full = kv["full_attention"][:2]
    assert k_full.shape[2] == S and k_full.shape[0] == 1


def test_g4_rollback_replays_to_sequential_state():
    lm = _g4_lm()
    prompt = _ids(*PROMPT)
    t1, t2, g2, g3 = 5, 6, 7, 9

    seq_cache = lm.make_cache()
    mx.eval(lm(prompt, cache=seq_cache).logits)
    seq_after_t1 = lm(_ids(t1), cache=seq_cache).logits
    seq_after_t2 = lm(_ids(t2), cache=seq_cache).logits

    ver_cache = lm.make_cache()
    mx.eval(lm(prompt, cache=ver_cache).logits)
    ver = lm(_ids(t1, g2, g3), cache=ver_cache, return_hidden=True)
    assert _close(ver.logits[:, :1], seq_after_t1, atol=REPLAY_ATOL)

    lm.rollback_speculative_cache(ver_cache, None, 0, 3)
    ver_after_t2 = lm(_ids(t2), cache=ver_cache).logits
    assert _close(ver_after_t2, seq_after_t2, atol=REPLAY_ATOL)


def test_g4_chunked_prefill_policy_table():
    lm = _g4_lm()
    draft = object()
    assert lm.chunked_prefill_policy(draft_model=None, prefill_kwargs={}) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp", prefill_kwargs={}) is False
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp",
        prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
    ) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="dflash",
        prefill_kwargs={"capture_layer_ids": [1]}) is False
    lm.no_chunked_prefill = True
    assert lm.chunked_prefill_policy(draft_model=None, prefill_kwargs={}) is False
