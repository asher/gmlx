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

pytest.importorskip("gmlx._vendor.mlx_vlm")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

import gmlx.load.loader as loader  # noqa: E402
import gmlx.load.mtp_target as mtp_target  # noqa: E402
import gmlx.load.wire as wire  # noqa: E402


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
    LanguageModel, _build = mtp_target._mtp_target_classes(model_type)
    missing = [h for h in mtp_target._MTP_TARGET_HOOKS if not hasattr(LanguageModel, h)]
    assert not missing, (
        f"mlx-vlm {model_type} LanguageModel missing hooks {missing}; "
        f"the MTP engine needs all of {mtp_target._MTP_TARGET_HOOKS}")


def test_mtp_target_resolver_rejects_unknown_arch():
    with pytest.raises(NotImplementedError):
        mtp_target._mtp_target_classes("llama")


# seam 3: the scans of the owned qwen3.5 gated_delta
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
    vgd = importlib.import_module("gmlx.models.qwen35.gated_delta")
    missing = [s for s in _SEAM3_SYMS if not hasattr(vgd, s)]
    assert not missing, f"mlx-vlm gated_delta missing seam-3 symbols {missing}"


def test_seam3_tiled_state_paths_run_the_ops(monkeypatch):
    """The state-capture and state-replay kernels pair heads in the grouped
    order only, so a tiled call takes the ops."""
    vgd = importlib.import_module("gmlx.models.qwen35.gated_delta")

    def _no_kernel(*a, **k):
        raise AssertionError("a tiled state path reached a grouped kernel")

    for s in _SEAM3_SYMS:
        if s.endswith(("_kernel", "_kernel_masked")) and s != "gated_delta_kernel":
            monkeypatch.setattr(vgd, s, _no_kernel)
    B, T, Hk, Hv, D = 1, 2, 2, 4, 32
    mx.random.seed(0)
    q = mx.random.normal((B, T, Hk, D)) * 0.2
    k = mx.random.normal((B, T, Hk, D)) * 0.2
    v = mx.random.normal((B, T, Hv, D))
    a, b = mx.random.normal((B, T, Hv)), mx.random.normal((B, T, Hv))
    A_log, dt_bias = mx.random.normal((Hv,)), mx.random.normal((Hv,))
    state = mx.zeros((B, Hv, D, D))
    steps = mx.array([T], dtype=mx.int32)
    for mask in (None, mx.ones((B, T), dtype=mx.bool_)):
        out = vgd.gated_delta_update_with_states(
            q, k, v, a, b, A_log, dt_bias, state, mask, tiled=True)
        new = vgd.gated_delta_state_update(
            k, v, a, b, A_log, dt_bias, state, steps, mask, tiled=True)
        mx.eval(out, new)
        assert mx.allclose(out[1], new, atol=1e-5).item()


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
    vgd = importlib.import_module("gmlx.models.qwen35.gated_delta")
    out = vgd._gated_delta_state_ops(k, v, g, beta, state, steps, tiled=True)
    # Each V head's resulting state row embeds the K head it was paired with.
    # hv=2 must match hv=0 (both -> k0) and hv=3 match hv=1 (both -> k1).
    s = out[0]  # [Hv, Dv, Dk]
    assert mx.allclose(s[0], s[2]).item(), "tiled: V head 2 should reuse K head 0"
    assert mx.allclose(s[1], s[3]).item(), "tiled: V head 3 should reuse K head 1"
    assert not mx.allclose(s[0], s[1]).item(), "K heads 0/1 must differ"


# drafter remap coverage (opt-in integration)
def _find_mtp_gguf(gguf_dir):
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.preflight import preflight
    for path in sorted(gguf_dir.rglob("*.gguf")):
        try:
            pf = preflight(str(path))
        except Exception:
            continue
        try:
            _a, _k, _am, meta, shapes = wire.load_gguf_wire_bytes(
                str(path), shards=pf.shards)
            cfg = synthesize_config(meta, shapes)
        except Exception:
            continue
        if int(cfg.get("mtp_num_hidden_layers", 0)) >= 1:
            return str(path), pf.arch
    return None, None


def test_mtp_drafter_remap_full_coverage(gguf_dir, cpu_device):
    """Every drafter param has a remapped source tensor of matching shape."""
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.preflight import preflight

    path, arch = _find_mtp_gguf(gguf_dir)
    if path is None:
        pytest.skip("no native-head MTP GGUF found in KQUANT_TEST_GGUF_DIR")

    pf = preflight(path)
    arrays, kqm, _am, meta, shapes = wire.load_gguf_wire_bytes(
        path, shards=pf.shards)
    config = synthesize_config(meta, shapes)

    cfg_mod = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.config")
    drafter_mod = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp")
    mtp_config = cfg_mod.Qwen3_5MTPConfig.from_dict(
        {"model_type": "qwen3_5_mtp", "text_config": dict(config)})
    drafter = drafter_mod.Qwen3_5MTPDraftModel(mtp_config)

    import gmlx.load.gguf_meta as gguf_meta
    n_head = gguf_meta.read_int(meta, f"{arch}.attention.head_count")
    n_head_kv = gguf_meta.first_nonzero_int(
        meta, f"{arch}.attention.head_count_kv")
    d_w, d_m, _stats = wire.remap_mtp_arrays(
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
    import gmlx.spec.mtp_drafter as md

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
    # prompt (~30 GB at 32k), so memory depends on the loop; this
    # pins its slicing arithmetic. Numerical equivalence across a real chunk
    # boundary is covered by the model-gated prefill-parity suite.
    import gmlx.spec.mtp_drafter as md

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
