#!/usr/bin/env python3
"""MTP (multi-token-prediction) speculative-decode plumbing.

Pure-logic tests pin the mlx-vlm contract the gmlx MTP path leans on, so a
mlx-vlm version bump that renames/drops a hook or a gated_delta symbol fails
loudly here instead of silently corrupting speculative decode. The drafter
remap-coverage check is an opt-in integration test (needs a native-head MTP
GGUF in ``KQUANT_TEST_GGUF_DIR``).
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from gmlx import gdn_patches, loader  # noqa: E402


@pytest.fixture
def cpu_device():
    # CPU numerics by design; restore so the flip never leaks to other tests.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


# target hook contract (version tripwire)
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_mtp_target_exposes_speculative_hooks(model_type):
    LanguageModel, _build = loader._mtp_target_classes(model_type)
    missing = [h for h in loader._MTP_TARGET_HOOKS if not hasattr(LanguageModel, h)]
    assert not missing, (
        f"mlx-vlm {model_type} LanguageModel missing hooks {missing}; "
        f"the MTP engine needs all of {loader._MTP_TARGET_HOOKS}")


def test_mtp_target_resolver_rejects_unknown_arch():
    with pytest.raises(NotImplementedError):
        loader._mtp_target_classes("llama")


# seam 3: mlx-vlm gated_delta tiled-V patch
_SEAM3_SYMS = (
    "gated_delta_ops",
    "gated_delta_kernel",
    "_gated_delta_with_states_ops",
    "_gated_delta_state_ops",
    "_gated_delta_with_states_kernel",
    "_gated_delta_with_states_kernel_masked",
    "_gated_delta_state_kernel",
    "_gated_delta_state_kernel_masked",
)


def test_mlxvlm_gated_delta_has_seam3_symbols():
    vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    missing = [s for s in _SEAM3_SYMS if not hasattr(vgd, s)]
    assert not missing, f"mlx-vlm gated_delta missing seam-3 symbols {missing}"


def test_seam3_patch_routes_state_paths_to_tiled_ops():
    """The patch must disable the grouped state kernels and swap in tiled ops,
    so the small-T MTP state-capture paths take the corrected fallback."""
    vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    saved = {s: getattr(vgd, s) for s in _SEAM3_SYMS}
    try:
        # the mlx-vlm patch depends on the mlx-lm one having run first.
        gdn_patches._patch_gated_delta_tiled_v()
        gdn_patches._patch_mlxvlm_gated_delta_tiled_v()
        assert vgd._gated_delta_with_states_kernel is None
        assert vgd._gated_delta_with_states_kernel_masked is None
        assert vgd._gated_delta_state_kernel is None
        assert vgd._gated_delta_state_kernel_masked is None
        assert vgd._gated_delta_with_states_ops is gdn_patches._tiled_gd_with_states_ops
        assert vgd._gated_delta_state_ops is gdn_patches._tiled_gd_state_ops
    finally:
        for s, v in saved.items():
            setattr(vgd, s, v)


def test_tiled_state_ops_uses_modulo_kv_mapping(cpu_device):
    """Tiled K->V mapping: with Hk=2, Hv=4, V head hv reads K head hv % Hk
    (i.e. [k0 k1 k0 k1]) - not the grouped hv // (Hv/Hk) ([k0 k0 k1 k1])."""
    B, T, Hk, Hv, Dk, Dv = 1, 1, 2, 4, 3, 3
    # Distinct K per head so the mapping is observable; V/g/beta neutralised.
    k = mx.reshape(mx.arange(Hk * Dk, dtype=mx.float32), (1, 1, Hk, Dk)) + 1.0
    v = mx.ones((B, T, Hv, Dv))
    g = mx.zeros((B, T, Hv))          # decay 0 -> state starts clean each step
    beta = mx.ones((B, T, Hv))
    state = mx.zeros((B, Hv, Dv, Dk))
    steps = mx.array([T], dtype=mx.int32)
    out = gdn_patches._tiled_gd_state_ops(k, v, g, beta, state, steps)
    # Each V head's resulting state row embeds the K head it was paired with.
    # hv=2 must match hv=0 (both -> k0) and hv=3 match hv=1 (both -> k1).
    s = out[0]  # [Hv, Dv, Dk]
    assert mx.allclose(s[0], s[2]).item(), "tiled: V head 2 should reuse K head 0"
    assert mx.allclose(s[1], s[3]).item(), "tiled: V head 3 should reuse K head 1"
    assert not mx.allclose(s[0], s[1]).item(), "K heads 0/1 must differ"


# drafter remap coverage (opt-in integration)
def _find_mtp_gguf(gguf_dir):
    from gmlx.config_synth import synthesize_config
    from gmlx.preflight import preflight
    for path in sorted(gguf_dir.rglob("*.gguf")):
        try:
            pf = preflight(str(path))
        except Exception:
            continue
        try:
            _a, _k, _am, meta, shapes = loader.load_gguf_wire_bytes(
                str(path), shards=pf.shards)
            cfg = synthesize_config(meta, shapes)
        except Exception:
            continue
        if int(cfg.get("mtp_num_hidden_layers", 0)) >= 1:
            return str(path), pf.arch
    return None, None


def test_mtp_drafter_remap_full_coverage(gguf_dir, cpu_device):
    """Every drafter param has a remapped source tensor of matching shape."""
    from gmlx.config_synth import synthesize_config
    from gmlx.preflight import preflight

    path, arch = _find_mtp_gguf(gguf_dir)
    if path is None:
        pytest.skip("no native-head MTP GGUF found in KQUANT_TEST_GGUF_DIR")

    pf = preflight(path)
    arrays, kqm, _am, meta, shapes = loader.load_gguf_wire_bytes(
        path, shards=pf.shards)
    config = synthesize_config(meta, shapes)

    cfg_mod = importlib.import_module(
        "mlx_vlm.speculative.drafters.qwen3_5_mtp.config")
    drafter_mod = importlib.import_module(
        "mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp")
    mtp_config = cfg_mod.Qwen3_5MTPConfig.from_dict(
        {"model_type": "qwen3_5_mtp", "text_config": dict(config)})
    drafter = drafter_mod.Qwen3_5MTPDraftModel(mtp_config)

    from gmlx import gguf_meta
    n_head = gguf_meta.read_int(meta, f"{arch}.attention.head_count")
    n_head_kv = gguf_meta.first_nonzero_int(
        meta, f"{arch}.attention.head_count_kv")
    d_w, d_m, _stats = loader.remap_mtp_arrays(
        arrays, kqm, arch,
        first_mtp_block=int(config["num_hidden_layers"]),
        num_mtp_layers=int(config.get("mtp_num_hidden_layers", 1)),
        n_head=n_head, n_head_kv=n_head_kv)

    n_replaced = loader.install_kquant_modules(drafter, d_m)
    assert n_replaced == len(d_m)

    drafter.eval()
    params = dict(tree_flatten(drafter.parameters()))
    missing = [p for p in params if p not in d_w]
    mismatch = [
        p for p, a in params.items()
        if p in d_w and tuple(d_w[p].shape) != tuple(a.shape)
    ]
    assert not missing, f"drafter params with no source: {missing}"
    assert not mismatch, f"shape mismatches: {mismatch}"


def test_seed_chunk_follows_prefill_step(monkeypatch):
    # Unset, the TF seed chunk tracks the serve prefill chunk so one knob
    # caps both attention transients; GMLX_HEAD_SEED_CHUNK still wins.
    from gmlx import mtp_drafter as md

    monkeypatch.setattr(md, "_SEED_CHUNK", None)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    assert md._seed_chunk() == 2048
    monkeypatch.setenv("PREFILL_STEP_SIZE", "512")
    assert md._seed_chunk() == 512
    monkeypatch.setattr(md, "_SEED_CHUNK", 1024)
    assert md._seed_chunk() == 1024


@pytest.mark.parametrize("chunk", [1, 3, 4, 1024])
@pytest.mark.parametrize("h_len,n", [(8, 8), (5, 8), (1, 8)])
def test_prefill_from_target_hidden_chunked_feed(monkeypatch, chunk, h_len, n):
    # The head seed must feed the SAME (token, hidden) pairs in the same
    # order regardless of chunk size, and seed from the final position. A
    # one-shot full-length pass materializes attention quadratically in the
    # prompt (~30 GB at 32k), so the loop is load-bearing for memory; this
    # pins its slicing arithmetic. Numerical equivalence across a real chunk
    # boundary is covered by the model-gated prefill-parity suite.
    from gmlx import mtp_drafter as md

    monkeypatch.setattr(md, "_SEED_CHUNK", chunk)
    D = 4

    class _Probe:
        prefill_from_target_hidden = md.QwenMTPDrafter.prefill_from_target_hidden
        _seed_profile = md.QwenMTPDrafter._seed_profile

        def __init__(self):
            self.tokens, self.hidden, self.seed = [], [], None
            self._cache = []  # the inter-chunk state eval reads it

        def _forward(self, tokens, hidden):
            assert tokens.shape[1] == hidden.shape[1] > 0
            self.tokens.append(tokens)
            self.hidden.append(hidden)
            return hidden  # identity: seed must equal the last hidden row

        def _set_seed(self, h, sampler, greedy):
            self.seed = h

    ids = mx.arange(n)[None, :].astype(mx.int32)
    hid = (mx.arange(n * D).reshape(1, n, D).astype(mx.float32) + 100.0)
    hid = hid[:, n - h_len:, :]  # engine may capture a truncated tail
    probe = _Probe()
    probe.prefill_from_target_hidden(ids, hid, 777, sampler=None)

    fed_tokens = mx.concatenate(probe.tokens, axis=1)
    fed_hidden = mx.concatenate(probe.hidden, axis=1)
    want_tokens = mx.concatenate(
        [ids[:, n - h_len + 1:], mx.array([[777]], dtype=mx.int32)], axis=1)
    assert fed_tokens.tolist() == want_tokens.tolist()
    assert fed_hidden.tolist() == hid.tolist()
    assert probe.seed.tolist() == hid[:, -1:, :].tolist()
    n_chunks = (h_len + chunk - 1) // chunk
    assert len(probe.tokens) == n_chunks
