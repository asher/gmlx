"""A real Qwen-Image 2.1 DiT GGUF through load_dit: one KQuantLinear per
quantized matrix, the fp32 norm leaves, and one small prefill and cached
step with and without a reference image, whose prefix cache has the bytes
the memory plan counts. Skipped unless KQUANT_TEST_GGUF_DIR holds a
Qwen-Image 2.1 DiT GGUF, which the tests find by its tensors.

Check 1 of the acceptance gate on the file the committed header inventory
was read from: its header is the inventory, the fingerprint names the
family and the fused layout, the synthesized config equals the official
one, and the strict load consumes every tensor as views of the file."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from image_fixtures import Q_HEADER, image_dits, load_json, official_config
from oracle_fixture import fx

import mlx.core as mx
from mlx.utils import tree_flatten

pytestmark = pytest.mark.integration

FLOATS = ("F32", "F16", "BF16")
# Tensors of the Q4_K_M file by (codec, dimensions).
Q4_K_M_CODECS = {("Q4_K", 2): 163, ("Q5_K", 2): 28, ("Q8_0", 2): 4,
                 ("Q6_K", 2): 1, ("F32", 1): 64, ("F32", 2): 1,
                 ("BF16", 2): 3, ("BF16", 1): 1}
# What the synthesized config holds beside the official keys.
OWN_KEYS = ["ffn_dim", "inner_dim", "mlp_layout", "modulation_dim",
            "num_train_timesteps", "rope_theta", "timestep_dim"]


def _qwen_file(gguf_dir) -> tuple[str, object]:
    from gmlx.image import fingerprint as fp
    from gmlx.load.headerscan import scan_gguf

    for p in image_dits(gguf_dir, "qwen_image21"):
        return p, fp.identify(fp.view_from_scans([scan_gguf(p)]))
    pytest.skip("no Qwen-Image 2.1 DiT GGUF under KQUANT_TEST_GGUF_DIR")


def _inventory_file(gguf_dir) -> tuple[str, dict]:
    """The file the committed header inventory was read from."""
    want = load_json("manifest.json")[Q_HEADER]
    for p in image_dits(gguf_dir, "qwen_image21"):
        if Path(p).name == want["file"] and os.path.getsize(p) == want["size"]:
            return p, want
    pytest.skip(f"{want['file']} is not under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def load(gguf_dir):
    path, ident = _qwen_file(gguf_dir)
    from gmlx.image.loader import load_dit

    lines: list = []
    model, info = load_dit(path, log=lambda *a: lines.append(
        " ".join(str(x) for x in a)))
    return ident, model, info, "\n".join(lines)


@pytest.fixture(scope="module")
def loaded(load):
    return load[:3]


def test_check1_the_real_header_is_the_committed_inventory(gguf_dir):
    from gmlx.image import fingerprint as fp
    from gmlx.load.headerscan import scan_gguf

    path, want = _inventory_file(gguf_dir)
    assert fx.file_identity(path)["sha256"] == want["sha256"]
    scan = scan_gguf(path)
    # The file has no KV, so no arch names it: the tensors do.
    assert not scan.truncated and not scan.kv
    view = fp.view_from_scans([scan])
    inventory = load_json(Q_HEADER)
    assert ({n: (tuple(t.shape), t.codec) for n, t in view.tensors.items()}
            == {n: (tuple(shape), codec)
                for n, shape, codec in inventory["tensors"]})
    assert Counter((t.codec, len(t.shape))
                   for t in view.tensors.values()) == Q4_K_M_CODECS
    got = fp.identify(view)
    assert (got.family.name, got.variant, got.layout, got.prefix) == (
        "qwen_image21", "default", "fused", "model.diffusion_model.")
    official = official_config("qwen_image21_transformer_config.json")
    assert {k: got.config[k] for k in official} == official
    assert sorted(set(got.config) - set(official)) == OWN_KEYS
    assert (got.config["ffn_dim"], got.config["inner_dim"],
            got.config["modulation_dim"]) == (12288, 4096, 16384)


def test_check1_the_strict_load_consumes_every_tensor_as_views(load):
    from mlx_kquant.nn import KQuantLinear

    from gmlx.image import memfit

    ident, model, info, log = load
    params = dict(tree_flatten(model.parameters()))
    tensors = ident.view.tensors
    assert f"loaded {len(params)} / {len(params)} model parameters" in log
    assert f"[verify] zero-copy views OK ({len(params)} params)" in log
    # One K-quant leaf per quantized matrix, in the codec of its tensor.
    leaves = {path: m for path, m in model.named_modules()
              if isinstance(m, KQuantLinear)}
    quant = {n: t for n, t in tensors.items() if t.codec not in FLOATS}
    assert all(len(t.shape) == 2 for t in quant.values())
    assert {f"{path}.weight" for path in leaves} == set(quant)
    assert Counter(str(m.kquant_type).upper() for m in leaves.values()) == \
        Counter(t.codec for t in quant.values())
    assert info.n_kquant == len(leaves) == len(quant)
    # Every other tensor is the parameter of its name, in its shape.
    plain = {n: t for n, t in tensors.items() if n not in quant}
    for name, t in plain.items():
        assert tuple(params[name].shape) == tuple(reversed(t.shape)), name
    assert ({n for n in params if n.rsplit(".", 1)[0] not in leaves}
            == {n for n in plain if n.rsplit(".", 1)[0] not in leaves})
    # The parameters hold the bytes of the file: no weight is dequantized
    # at the load. The casts the load owns are the BF16 norm weights, and
    # each doubles its bytes.
    cast = [n for n, t in plain.items() if "norm" in n and t.codec != "F32"]
    assert info.n_owned_casts == len(cast)
    grown = sum(memfit.tensor_bytes(tensors[n].shape, tensors[n].codec)
                for n in cast)
    assert sum(int(v.nbytes) for v in params.values()) == (
        memfit.file_bytes(ident.view) + grown)
    # The fused layout: one gate and up product and the out product.
    assert ident.layout == "fused" and model.cfg.fused_mlp
    for blk in model.transformer_blocks:
        assert sorted(k for k, _ in blk.img_mlp.named_modules() if k) == [
            "gate_up", "out"]


def test_quantized_leaf_count_and_fp32_norms(loaded):
    ident, model, info = loaded
    floats = ("F32", "F16", "BF16")
    tensors = ident.view.tensors
    quant = sum(1 for n, t in tensors.items() if len(t.shape) == 2
                and t.codec not in floats and "norm" not in n)
    assert info.n_kquant == quant
    assert info.n_owned_casts == sum(1 for n, t in tensors.items()
                                     if "norm" in n and t.codec != "F32")
    assert model.act_dtype == mx.bfloat16
    assert len(model.transformer_blocks) == 32
    blk = model.transformer_blocks[0]
    for leaf in (model.txt_in.text_norm, model.norm_out.linear,
                 blk.attn.norm_q, blk.attn.norm_k):
        assert leaf.weight.dtype == mx.float32


# name: (text rows, image slots, reference shape in latent tokens)
STEPS = {"text": (24, (), ()),
         "ref1": (40, tuple(range(10, 18)), ((4, 8),))}


@pytest.mark.parametrize("name", list(STEPS))
def test_one_small_cached_step_is_finite(loaded, name):
    from gmlx.image import memfit
    from gmlx.image.conditioning import Conditioning, array_bytes
    from gmlx.image.families.qwen_image21 import model as qm

    _, model, _ = loaded
    rows, slots, refs = STEPS[name]
    cfg = model.cfg
    rng = np.random.default_rng(0)

    def normal(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    cond = Conditioning(
        normal(rows, cfg.context_dim), name,
        ref_latents=tuple(normal(cfg.in_channels, h, w) for h, w in refs),
        image_slots=slots)
    prefix = qm.prefill_prefix(model, cond)
    tokens = rows - len(slots) + sum(h * w for h, w in refs)
    assert prefix.tokens == tokens
    row = memfit.PEAKS["qwen_image21"].prefix
    assert array_bytes(prefix.arrays()) == row.cache_bytes(tokens, 2)
    lat = normal(1, cfg.in_channels, 16, 16)
    outs = [qm.forward_target(model, lat, s, cond, prefix)
            for s in (0.8, 0.3)]
    mx.eval(outs)
    for out in outs:
        assert out.shape == lat.shape and out.dtype == mx.float32
        assert bool(mx.all(mx.isfinite(out)))
    assert not np.array_equal(np.array(outs[0]), np.array(outs[1]))
