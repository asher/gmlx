"""Hadamard-folded GGUFs: header parsing, target resolution, the rotation
math and the module swap.

The parser is checked against every rule the reference loader applies, on
dicts and on a minted GGUF read through both header readers. The MLX
rotation forms are checked against a float64 numpy oracle built from the
explicit popcount-parity matrix. The module swap runs on a tiny tree with
q8_0 wire, the same shape the loader produces.
"""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq

from gmlx.load.hadamard import (
    HADAMARD_ARCHES,
    FoldTarget,
    HadamardSpecError,
    hadamard_matrix,
    hadamard_targets_for,
    parse_hadamard_spec,
    permute_grouped,
    reference_rotate,
    reference_rotate_inverse,
    resolve_hadamard_targets,
)
from gmlx.load.hadamard_modules import (
    HadamardKQuantEmbedding,
    HadamardKQuantLinear,
    _Fold,
    install_hadamard_modules,
    reset_rotation_count,
    rotate,
    rotate_inverse,
    rotation_count,
)
from gmlx.load.modules import install_kquant_modules
from mlx_kquant.nn import KQuantEmbedding, KQuantLinear

BLOCK = 64
WIDTHS = (256, 384)


def _signs(rng, width):
    return rng.choice(np.array([-1, 1], dtype=np.int64), size=width)


def _spec_kv(rng, *, sign_mode="explicit", widths=WIDTHS, **over):
    kv = {
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": BLOCK,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": sign_mode,
        "prism.hadamard.weight_names": [
            "output.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_output.weight",
        ],
        "prism.hadamard.inverse_weight_names": ["token_embd.weight"],
        "prism.hadamard.gdn_v_grouped": True,
    }
    if sign_mode == "explicit":
        kv["prism.hadamard.sign_widths"] = list(widths)
        kv["prism.hadamard.sign_values"] = np.concatenate(
            [_signs(rng, w) for w in widths]).tolist()
    for k, v in over.items():
        if v is None:
            kv.pop("prism.hadamard." + k, None)
        else:
            kv["prism.hadamard." + k] = v
    return kv


def _qwen35_meta(kv):
    meta = dict(kv)
    meta["general.architecture"] = "qwen35"
    meta["qwen35.ssm.time_step_rank"] = 6
    meta["qwen35.ssm.group_count"] = 2
    return meta


# GGUF shapes are stored innermost-first, so shape[0] is the input width.
_SHAPES = {
    "output.weight": (256, 32),
    "blk.0.attn_qkv.weight": (256, 128),
    "blk.0.attn_gate.weight": (256, 384),
    "blk.0.ssm_out.weight": (384, 256),
    "blk.0.ffn_gate.weight": (256, 512),
    "blk.0.ffn_up.weight": (256, 512),
    "blk.0.ffn_down.weight": (384, 256),
    "blk.1.attn_q.weight": (256, 256),
    "blk.1.attn_output.weight": (384, 256),
    "token_embd.weight": (256, 32),
}


def test_no_header_means_no_fold():
    assert parse_hadamard_spec({"general.architecture": "qwen35"}) is None
    assert hadamard_targets_for({}, "qwen35", {}) == {}


def test_explicit_spec_parses():
    rng = np.random.default_rng(0)
    kv = _spec_kv(rng)
    spec = parse_hadamard_spec(kv)
    assert spec.block == BLOCK
    assert sorted(spec.signs) == sorted(WIDTHS)
    for w in WIDTHS:
        assert spec.signs[w].shape == (w,)
        assert set(np.unique(spec.signs[w])) <= {-1, 1}
    assert spec.weight_names == tuple(kv["prism.hadamard.weight_names"])
    assert spec.inverse_names == ("token_embd.weight",)
    assert spec.gdn_v_grouped is True
    flat = np.concatenate([spec.signs[w] for w in WIDTHS])
    assert flat.tolist() == kv["prism.hadamard.sign_values"]


def test_identity_spec_has_no_signs():
    rng = np.random.default_rng(0)
    spec = parse_hadamard_spec(_spec_kv(rng, sign_mode="identity"))
    assert spec.signs is None
    assert spec.gdn_v_grouped is True


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"version": 2}, "version"),
        ({"block_size": 48}, "block_size"),
        ({"block_size": None}, "block_size is missing"),
        ({"transform": "walsh"}, "transform"),
        ({"axis": "input-first-dimension"}, "axis"),
        ({"sign_mode": "random"}, "sign_mode"),
        ({"weight_names": []}, "weight_names is empty"),
        ({"weight_names": None}, "weight_names is missing"),
        ({"sign_widths": []}, "sign_widths is empty"),
        ({"sign_widths": [256, 200]}, "sign width"),
        ({"sign_widths": [256, 384, 64]}, "sign width"),
        ({"sign_widths": [256]}, "length mismatch"),
        ({"sign_values": [1] * 639 + [2]}, "sign values must be"),
        ({"weight_names": ["blk.0.attn_norm.weight"]}, "Hadamard-aware"),
        ({"weight_names": ["blk.0.ffn_up.weight", "blk.0.ffn_up.weight"]},
         "duplicate"),
        ({"inverse_weight_names": ["output.weight"]}, "inverse-after-lookup"),
        ({"inverse_weight_names": ["token_embd.weight", "token_embd.weight"]},
         "duplicate"),
    ],
)
def test_reference_checks_raise(over, needle):
    rng = np.random.default_rng(0)
    with pytest.raises(HadamardSpecError, match=needle):
        parse_hadamard_spec(_spec_kv(rng, **over))


def test_inverse_name_cannot_also_be_folded():
    rng = np.random.default_rng(0)
    kv = _spec_kv(rng)
    kv["prism.hadamard.weight_names"] = ["token_embd.weight"]
    with pytest.raises(HadamardSpecError, match="Hadamard-aware"):
        parse_hadamard_spec(kv)


def test_minted_gguf_reads_the_same_through_both_readers(tmp_path):
    from gguf import GGUFWriter

    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.wire import load_gguf_wire_bytes

    rng = np.random.default_rng(1)
    kv = _spec_kv(rng, widths=(256,))
    w = GGUFWriter(str(tmp_path / "fold.gguf"), "qwen35")
    for key, val in kv.items():
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, int):
            w.add_uint32(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        else:
            w.add_array(key, val)
    w.add_tensor("token_embd.weight", np.zeros((32, 256), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    path = str(tmp_path / "fold.gguf")

    scan = scan_gguf(path, array_limit=32768)
    from_scan = parse_hadamard_spec(scan.kv)
    _, _, _, meta, _ = load_gguf_wire_bytes(path, expect_quant=False)
    from_wire = parse_hadamard_spec(meta)
    assert from_scan.block == from_wire.block == BLOCK
    assert from_scan.weight_names == from_wire.weight_names
    assert from_scan.inverse_names == from_wire.inverse_names == ("token_embd.weight",)
    assert from_scan.gdn_v_grouped is from_wire.gdn_v_grouped is True
    np.testing.assert_array_equal(from_scan.signs[256], from_wire.signs[256])
    assert from_scan.signs[256].tolist() == kv["prism.hadamard.sign_values"]


def test_targets_resolve_to_module_paths():
    rng = np.random.default_rng(0)
    meta = _qwen35_meta(_spec_kv(rng))
    targets = hadamard_targets_for(meta, "qwen35", _SHAPES)
    assert set(targets) == {
        "lm_head",
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.o_proj",
        "model.embed_tokens",
    }
    out = targets["model.layers.0.linear_attn.out_proj"]
    assert out.width == 384 and out.block == BLOCK and not out.inverse
    # n_v=6, n_k=2 over width 384: rep 3, nk 2, hd 64.
    assert out.perm == (3, 2, 64)
    down = targets["model.layers.0.mlp.down_proj"]
    assert down.width == 384 and down.perm is None
    np.testing.assert_array_equal(down.signs, out.signs)
    emb = targets["model.embed_tokens"]
    assert emb.inverse and emb.width == 256 and emb.perm is None
    np.testing.assert_array_equal(emb.signs, targets["lm_head"].signs)
    for t in targets.values():
        assert t.signs.shape == (t.width,)

    prefixed = hadamard_targets_for(
        meta, "qwen35", _SHAPES, target_prefix="language_model")
    assert "language_model.model.embed_tokens" in prefixed
    assert "language_model.lm_head" in prefixed


def test_targets_without_grouping_have_no_perm():
    rng = np.random.default_rng(0)
    meta = _qwen35_meta(_spec_kv(rng, gdn_v_grouped=False))
    targets = hadamard_targets_for(meta, "qwen35", _SHAPES)
    assert all(t.perm is None for t in targets.values())


def test_grouping_needs_the_gdn_head_counts():
    rng = np.random.default_rng(0)
    meta = _qwen35_meta(_spec_kv(rng))
    meta["qwen35.ssm.group_count"] = 4
    with pytest.raises(HadamardSpecError, match="group count"):
        hadamard_targets_for(meta, "qwen35", _SHAPES)
    meta["qwen35.ssm.group_count"] = 2
    meta["qwen35.ssm.time_step_rank"] = 10
    with pytest.raises(HadamardSpecError, match="V-head count"):
        hadamard_targets_for(meta, "qwen35", _SHAPES)


def test_arch_outside_allowlist_is_refused():
    rng = np.random.default_rng(0)
    assert "llama" not in HADAMARD_ARCHES
    spec = parse_hadamard_spec(_spec_kv(rng))
    with pytest.raises(HadamardSpecError, match="llama"):
        resolve_hadamard_targets(spec, "llama", {}, _SHAPES)


def test_expert_stack_is_refused():
    rng = np.random.default_rng(0)
    meta = _qwen35_meta(_spec_kv(rng, weight_names=["blk.0.ffn_up_exps.weight"]))
    shapes = dict(_SHAPES, **{"blk.0.ffn_up_exps.weight": (256, 512, 8)})
    with pytest.raises(NotImplementedError, match="expert stack"):
        hadamard_targets_for(meta, "qwen35", shapes)


def test_missing_sign_width_and_tensor_are_refused():
    rng = np.random.default_rng(0)
    meta = _qwen35_meta(_spec_kv(rng, widths=(256,)))
    with pytest.raises(HadamardSpecError, match="no sign vector"):
        hadamard_targets_for(meta, "qwen35", _SHAPES)
    meta = _qwen35_meta(_spec_kv(rng))
    shapes = {k: v for k, v in _SHAPES.items() if k != "blk.0.ssm_out.weight"}
    with pytest.raises(HadamardSpecError, match="not in the file"):
        hadamard_targets_for(meta, "qwen35", shapes)
    shapes = dict(_SHAPES, **{"blk.0.ssm_out.weight": (400, 256)})
    with pytest.raises(HadamardSpecError, match="multiple of the block"):
        hadamard_targets_for(meta, "qwen35", shapes)


def test_hadamard_matrix_is_orthogonal_and_natural_order():
    h = hadamard_matrix(8)
    np.testing.assert_allclose(h @ h, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(h, h.T)
    # Row 1 alternates sign: (-1)^popcount(1 & j) = (-1)^(j & 1).
    np.testing.assert_allclose(h[1] * np.sqrt(8), [1, -1, 1, -1, 1, -1, 1, -1])
    # mx.hadamard_transform is the same natural-order transform.
    x = np.random.default_rng(0).standard_normal((3, 1024)).astype(np.float32)
    got = np.array(mx.hadamard_transform(mx.array(x), scale=1024 ** -0.5))
    np.testing.assert_allclose(got, x @ hadamard_matrix(1024), atol=2e-5)


def test_permute_grouped_moves_tiled_heads_to_grouped_order():
    rep, nk, hd = 3, 2, 4
    x = np.arange(rep * nk * hd).reshape(rep, nk, hd)
    out = permute_grouped(x.reshape(1, -1), (rep, nk, hd)).reshape(nk, rep, hd)
    for r in range(rep):
        for k in range(nk):
            np.testing.assert_array_equal(out[k, r], x[r, k])


def _fold(target):
    signs = None if target.signs is None else mx.array(
        target.signs.astype(np.float32))
    return _Fold(target, signs)


@pytest.mark.parametrize("perm", [None, (3, 2, 64)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_rotate_matches_numpy_reference(perm, dtype):
    rng = np.random.default_rng(2)
    width = 384
    target = FoldTarget(width=width, block=BLOCK, signs=_signs(rng, width),
                        perm=perm)
    x = mx.array(rng.standard_normal((2, 5, width)).astype(np.float32)).astype(dtype)
    got = rotate(x, _fold(target))
    assert got.dtype == dtype and got.shape == x.shape
    ref = reference_rotate(np.array(x.astype(mx.float32)), target)
    tol = {mx.float32: 1e-5, mx.float16: 2e-2, mx.bfloat16: 1e-1}[dtype]
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), ref, atol=tol)


def test_rotate_identity_signs():
    rng = np.random.default_rng(3)
    target = FoldTarget(width=256, block=BLOCK, signs=None)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    got = np.array(rotate(mx.array(x), _fold(target)))
    np.testing.assert_allclose(got, reference_rotate(x, target), atol=1e-5)


def test_inverse_undoes_forward():
    rng = np.random.default_rng(4)
    target = FoldTarget(width=256, block=BLOCK, signs=_signs(rng, 256))
    inv = FoldTarget(width=256, block=BLOCK, signs=target.signs, inverse=True)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    np.testing.assert_allclose(
        reference_rotate_inverse(reference_rotate(x, target), inv), x, atol=1e-12)
    back = rotate_inverse(rotate(mx.array(x), _fold(target)), _fold(inv))
    np.testing.assert_allclose(np.array(back), x, atol=1e-5)
    ref = reference_rotate_inverse(x, inv)
    got = np.array(rotate_inverse(mx.array(x), _fold(inv)))
    np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("perm", [None, (3, 16, 128)])
def test_kernel_form_matches_ops_form(monkeypatch, perm, dtype):
    """With mlx-kquant's hadamard_rotate installed, the kernel and the
    MLX-op form agree to the output dtype's rounding, forward (with the
    head permute) and inverse."""
    if getattr(kq, "hadamard_rotate", None) is None:
        pytest.skip("installed mlx-kquant has no hadamard_rotate")
    if mx.default_device() != mx.gpu:
        pytest.skip("kernel form runs on the GPU device only")
    rng = np.random.default_rng(12)
    width = 6144
    target = FoldTarget(width=width, block=1024, signs=_signs(rng, width), perm=perm)
    inv = FoldTarget(width=width, block=1024, signs=target.signs, inverse=True)
    x = mx.array(rng.standard_normal((3, width)).astype(np.float32)).astype(dtype)
    monkeypatch.setenv("GMLX_HADAMARD_KERNEL", "1")
    k_fwd = rotate(x, _fold(target))
    k_inv = rotate_inverse(x, _fold(inv))
    monkeypatch.setenv("GMLX_HADAMARD_KERNEL", "0")
    o_fwd = rotate(x, _fold(target))
    o_inv = rotate_inverse(x, _fold(inv))
    tol = 2e-2 if dtype == mx.bfloat16 else 3e-3
    for k, o in ((k_fwd, o_fwd), (k_inv, o_inv)):
        assert k.dtype == dtype
        kf, of = np.array(k.astype(mx.float32)), np.array(o.astype(mx.float32))
        assert np.abs(kf - of).max() / np.abs(of).max() < tol


def test_rotation_switches(monkeypatch):
    rng = np.random.default_rng(5)
    target = FoldTarget(width=256, block=BLOCK, signs=_signs(rng, 256))
    x = mx.array(rng.standard_normal((2, 256)).astype(np.float32))
    monkeypatch.setenv("GMLX_HADAMARD_TRACE", "1")
    reset_rotation_count()
    rotate(x, _fold(target))
    rotate_inverse(x, _fold(target))
    assert rotation_count() == 2
    monkeypatch.setenv("GMLX_HADAMARD_ROTATE", "0")
    assert rotate(x, _fold(target)) is x
    assert rotate_inverse(x, _fold(target)) is x
    assert rotation_count() == 2


class _Block(nn.Module):
    def __init__(self, vocab, dims, out):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dims)
        self.proj = nn.Linear(dims, out, bias=False)
        self.plain = nn.Linear(dims, out, bias=False)


class _Outer(nn.Module):
    def __init__(self, vocab, dims, out):
        super().__init__()
        self.model = _Block(vocab, dims, out)


def _swapped_tree(vocab=16, dims=256, out=64):
    mx.random.seed(0)
    model = _Outer(vocab, dims, out)
    table_q, table_s = kq.quantize(mx.random.normal((vocab, dims)), "q8_0")
    proj_q, proj_s = kq.quantize(mx.random.normal((out, dims)), "q8_0")
    weights = {
        "model.embed_tokens.weight": table_q.reshape(vocab, -1),
        "model.embed_tokens.scales": table_s,
        "model.proj.weight": proj_q.reshape(out, -1),
        "model.proj.scales": proj_s,
        "model.plain.weight": mx.random.normal((out, dims)),
    }
    meta = {"model.embed_tokens.weight": "q8_0", "model.proj.weight": "q8_0"}
    assert install_kquant_modules(model, meta) == 2
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _targets(rng, dims=256):
    signs = _signs(rng, dims)
    return {
        "proj": FoldTarget(width=dims, block=BLOCK, signs=signs),
        "embed_tokens": FoldTarget(width=dims, block=BLOCK, signs=signs,
                                   inverse=True),
    }


def test_install_swaps_only_the_named_leaves():
    rng = np.random.default_rng(6)
    model = _swapped_tree()
    targets = _targets(rng)
    assert install_hadamard_modules(model, targets) == 2
    assert type(model.model.proj) is HadamardKQuantLinear
    assert type(model.model.embed_tokens) is HadamardKQuantEmbedding
    assert type(model.model.plain) is nn.Linear
    # The fold sits outside the parameter tree.
    assert "_hadamard" not in dict(model.parameters()).get("model", {}).get("proj", {})
    assert model.model.proj._hadamard.signs is model.model.embed_tokens._hadamard.signs
    # A second install finds the swapped classes and changes nothing.
    assert install_hadamard_modules(model, targets) == 0
    assert install_hadamard_modules(model, {}) == 0


def test_folded_linear_rotates_unless_pre_rotated():
    rng = np.random.default_rng(7)
    model = _swapped_tree()
    targets = _targets(rng)
    install_hadamard_modules(model, targets)
    proj = model.model.proj
    x = mx.array(rng.standard_normal((3, 256)).astype(np.float32)).astype(mx.bfloat16)
    plain = KQuantLinear.__call__(proj, x)
    rotated = KQuantLinear.__call__(proj, rotate(x, proj._hadamard))
    assert mx.array_equal(proj(x), rotated)
    assert mx.array_equal(proj(x, pre_rotated=True), plain)
    assert not mx.array_equal(plain, rotated)


def test_folded_embedding_unrotates_rows_and_rotates_the_tied_head():
    rng = np.random.default_rng(8)
    model = _swapped_tree()
    targets = _targets(rng)
    install_hadamard_modules(model, targets)
    emb = model.model.embed_tokens
    ids = mx.array([[1, 5], [9, 3]])
    rows = KQuantEmbedding.__call__(emb, ids)
    assert mx.array_equal(emb(ids), rotate_inverse(rows, emb._hadamard))
    assert emb(ids).dtype == rows.dtype
    x = mx.array(rng.standard_normal((2, 256)).astype(np.float32)).astype(mx.bfloat16)
    assert mx.array_equal(
        emb.as_linear(x), KQuantEmbedding.as_linear(emb, rotate(x, emb._hadamard)))


def test_install_refuses_bad_targets():
    rng = np.random.default_rng(9)
    targets = _targets(rng)
    model = _swapped_tree()
    with pytest.raises(ValueError, match="no module"):
        install_hadamard_modules(model, dict(targets, missing=targets["proj"]))
    model = _swapped_tree()
    with pytest.raises(TypeError, match="not KQuantLinear"):
        install_hadamard_modules(model, {"plain": targets["proj"]})
    model = _swapped_tree()
    wrong = FoldTarget(width=512, block=BLOCK, signs=_signs(rng, 512))
    with pytest.raises(ValueError, match="row width"):
        install_hadamard_modules(model, {"proj": wrong})


def test_occupancy_fuse_skips_folded_projections():
    from gmlx.upstream.occupancy_fuse import _same_codec_kquant

    rng = np.random.default_rng(10)
    model = _swapped_tree()
    assert _same_codec_kquant([model.model.proj, model.model.proj])
    install_hadamard_modules(model, _targets(rng))
    assert not _same_codec_kquant([model.model.proj, model.model.proj])


def test_lora_delta_uses_the_unrotated_input():
    from gmlx.load import modules

    rng = np.random.default_rng(11)
    model = _swapped_tree()
    install_hadamard_modules(model, _targets(rng))
    proj = model.model.proj
    a = mx.array(rng.standard_normal((2, 256)).astype(np.float32))
    b = mx.array(rng.standard_normal((64, 2)).astype(np.float32))
    wrap = modules.LoRAKQuantLinear(proj, a, b, 0.5)
    x = mx.array(rng.standard_normal((3, 256)).astype(np.float32)).astype(mx.bfloat16)
    delta = lambda v: (0.5 * ((v @ a.T) @ b.T)).astype(mx.bfloat16)  # noqa: E731
    got = np.array(wrap(x).astype(mx.float32))
    plain = np.array((proj(x) + delta(x)).astype(mx.float32))
    rotated = np.array((proj(x) + delta(rotate(x, proj._hadamard))).astype(mx.float32))
    # bf16 rounding of the delta path separates the two forms by far less
    # than the rotation does.
    assert np.abs(got - plain).max() < 0.1 * np.abs(got - rotated).max()
    np.testing.assert_allclose(got, plain, atol=0.5, rtol=5e-2)
