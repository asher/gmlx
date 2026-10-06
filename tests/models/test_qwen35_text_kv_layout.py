"""The K to V head order of a gated delta layer on the text route is the
model's own.

A Qwen3.5 GGUF stores its V heads tiled (V head hv reads K head hv % Hk)
and an MLX checkpoint grouped (hv // (Hv / Hk)). Both build the copied
mlx-lm classes, the MTP target of the GGUF builds the owned classes, whose
prefill scans through the copy's ``gated_delta``, and a Qwen3-Next model
of either format keeps the grouped order on the same scans. Each layer
reads ``kv_head_layout`` from its config and passes it to every scan, so
one process holds a model of each order: a distill run with an MLX
checkpoint beside a GGUF is one.

The model cases run in child processes that load a toy GGUF through
``gmlx.load_model``, a toy MLX checkpoint through the copy's
``load_model``, an MTP target as the MTP loader builds and arms it, and a
Qwen3-Next model as the loader builds it, in either order. Each model
gives the logits it gives as the only model of its process. The GGUF, the
checkpoint and the MTP target also give the logits of a reference of the
same format with one K head per V head, which holds the pairing in its
weights.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

_GPU = pytest.param("gpu", marks=pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")), reason="needs the Metal GPU"))
_DEVICES = (_GPU, "cpu")
_TILED = ("gguf", "mtp")
_GROUPED = ("mlx", "next")
_ORDERS = (_TILED + _GROUPED, _GROUPED + _TILED)
# The kinds that have a reference with one K head per V head.
_WITH_REFERENCE = ("gguf", "mtp", "mlx")

# Model against its reference: the same scan on rows that hold equal
# values. Model against the same model alone in its process: the same code
# on the same weights.
REF_ATOL = 1e-5
ALONE_ATOL = 1e-6
# Neither order may pass for the other.
ORDERS_DIFFER = 0.05

_COMMON = r"""
import json
import os
import sys

import numpy as np

H, LAYERS, FFN, HEADS, KV, HD = 64, 4, 128, 4, 2, 32
HK, HV, DK, DV, CONV = 2, 6, 32, 32, 4
INNER = HV * DV
SPECIALS = ["<s>", "</s>", "<pad>"]


def tokens():
    from tokenizers import pre_tokenizers
    return SPECIALS + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"]


def qkv_rows(layout):
    # Rows of a [q | k | v] tensor with one K head per V head, where K
    # head j is K head heads[j] of the tensor with HK of them.
    heads = [h % HK if layout == "tiled" else h // (HV // HK)
             for h in range(HV)]
    kd = HK * DK
    qk = np.concatenate([np.arange(h * DK, (h + 1) * DK) for h in heads])
    return np.concatenate([qk, kd + qk, 2 * kd + np.arange(INNER)])


def text_raw(hk, **over):
    raw = dict(
        model_type="qwen3_5_text", hidden_size=H, intermediate_size=FFN,
        linear_num_value_heads=HV, linear_num_key_heads=hk,
        linear_key_head_dim=DK, linear_value_head_dim=DV,
        linear_conv_kernel_dim=CONV, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, rms_norm_eps=1e-6,
        vocab_size=len(tokens()), num_key_value_heads=KV,
        max_position_embeddings=1024, tie_word_embeddings=False, head_dim=HD,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.25},
        full_attention_interval=4)
    raw.update(over)
    return raw
"""

_MINT = _COMMON + r"""
from gguf import GGUFWriter

out = sys.argv[1]


def mint_gguf(path, wide=None):
    # A qwen35 GGUF of float32 tensors. ``wide`` writes one K head per V
    # head, paired in that order.
    arch, toks = "qwen35", tokens()
    w = GGUFWriter(path, arch)
    w.add_uint32(f"{arch}.embedding_length", H)
    w.add_uint32(f"{arch}.block_count", LAYERS)
    w.add_uint32(f"{arch}.attention.head_count", HEADS)
    w.add_uint32(f"{arch}.attention.head_count_kv", KV)
    w.add_uint32(f"{arch}.attention.key_length", HD)
    w.add_uint32(f"{arch}.attention.value_length", HD)
    w.add_uint32(f"{arch}.feed_forward_length", FFN)
    w.add_uint32(f"{arch}.context_length", 1024)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(f"{arch}.full_attention_interval", 4)
    w.add_uint32(f"{arch}.rope.dimension_count", HD // 4)
    w.add_array(f"{arch}.rope.dimension_sections", [2, 1, 1, 0])
    w.add_float32(f"{arch}.rope.freq_base", 10000000.0)
    w.add_uint32(f"{arch}.ssm.conv_kernel", CONV)
    w.add_uint32(f"{arch}.ssm.group_count", HV if wide else HK)
    w.add_uint32(f"{arch}.ssm.inner_size", INNER)
    w.add_uint32(f"{arch}.ssm.state_size", DK)
    w.add_uint32(f"{arch}.ssm.time_step_rank", HV)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen35")
    w.add_array("tokenizer.ggml.tokens", toks)
    w.add_array("tokenizer.ggml.merges", ["H e", "w o"])
    w.add_array("tokenizer.ggml.token_type", [3, 3, 3] + [1] * (len(toks) - 3))
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    rng = np.random.default_rng(0)
    rows = qkv_rows(wide) if wide else None

    def t(name, *shape, scale=0.1, qkv=False):
        a = rng.standard_normal(shape).astype(np.float32) * scale
        if name.endswith("norm.weight"):
            a = 1.0 + a
        if name.endswith("ssm_a"):
            a = -np.exp(a)
        if qkv and rows is not None:
            a = np.ascontiguousarray(a[rows])
        w.add_tensor(name, a)

    t("token_embd.weight", len(toks), H)
    t("output_norm.weight", H)
    t("output.weight", len(toks), H)
    for i in range(LAYERS):
        t(f"blk.{i}.attn_norm.weight", H)
        t(f"blk.{i}.post_attention_norm.weight", H)
        if (i + 1) % 4:
            t(f"blk.{i}.attn_qkv.weight", 2 * HK * DK + INNER, H, qkv=True)
            t(f"blk.{i}.attn_gate.weight", INNER, H)
            t(f"blk.{i}.ssm_a", HV)
            t(f"blk.{i}.ssm_alpha.weight", HV, H)
            t(f"blk.{i}.ssm_beta.weight", HV, H)
            t(f"blk.{i}.ssm_conv1d.weight", 2 * HK * DK + INNER, CONV,
              scale=0.5, qkv=True)
            t(f"blk.{i}.ssm_dt.bias", HV)
            t(f"blk.{i}.ssm_norm.weight", DV)
            t(f"blk.{i}.ssm_out.weight", H, INNER)
        else:
            t(f"blk.{i}.attn_q.weight", HEADS * HD * 2, H)
            t(f"blk.{i}.attn_k.weight", KV * HD, H)
            t(f"blk.{i}.attn_v.weight", KV * HD, H)
            t(f"blk.{i}.attn_output.weight", H, HEADS * HD)
            t(f"blk.{i}.attn_q_norm.weight", HD)
            t(f"blk.{i}.attn_k_norm.weight", HD)
        t(f"blk.{i}.ffn_gate.weight", FFN, H)
        t(f"blk.{i}.ffn_up.weight", FFN, H)
        t(f"blk.{i}.ffn_down.weight", H, FFN)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


import mlx.core as mx
from mlx.utils import tree_flatten

from gmlx._vendor.mlx_lm.models import qwen3_5


def mint_mlx(path, wide=None):
    # An MLX checkpoint of the copy's qwen3_5 model, as mlx-lm saves one:
    # its config names no head order.
    os.makedirs(path)
    mx.random.seed(0)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(
        {"model_type": "qwen3_5", "text_config": text_raw(HK)}))
    weights = dict(tree_flatten(model.parameters()))
    if wide:
        idx = mx.array(qkv_rows(wide))
        for name in list(weights):
            if name.endswith(("linear_attn.in_proj_qkv.weight",
                              "linear_attn.conv1d.weight")):
                weights[name] = weights[name][idx]
    mx.save_safetensors(os.path.join(path, "model.safetensors"), weights)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5",
                   "text_config": text_raw(HV if wide else HK)}, f)


def mint_mmproj(path):
    # A clip GGUF with the projector type of a Qwen3.5 pair and the
    # metadata of a one-block tower. It holds one tensor: a load that is
    # refused before the weights install reads no more.
    w = GGUFWriter(path, "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen3vl_merger")
    w.add_uint32("clip.vision.embedding_length", 32)
    w.add_uint32("clip.vision.block_count", 1)
    w.add_uint32("clip.vision.attention.head_count", 2)
    w.add_uint32("clip.vision.feed_forward_length", 64)
    w.add_uint32("clip.vision.image_size", 56)
    w.add_uint32("clip.vision.patch_size", 14)
    w.add_uint32("clip.vision.projection_dim", H)
    w.add_uint32("clip.vision.spatial_merge_size", 2)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_tensor("v.post_ln.weight", np.ones(32, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


for name, wide in (("narrow", None), ("wide-tiled", "tiled"),
                   ("wide-grouped", "grouped")):
    mint_gguf(os.path.join(out, f"{name}.gguf"), wide)
    mint_mlx(os.path.join(out, f"mlx-{name}"), wide)
mint_mmproj(os.path.join(out, "mmproj.gguf"))
"""

_CHILD = _COMMON + r"""
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

root, order, device = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
references = sys.argv[4] == "references"
mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)

import gmlx
from gmlx._vendor.mlx_lm.models.cache import make_prompt_cache
from gmlx._vendor.mlx_lm.utils import load_model as load_checkpoint
from gmlx.load.loader import build_model
from gmlx.spec.mtp_load import _arm_spec_target


def gguf(name):
    return gmlx.load_model(str(root / f"{name}.gguf"), verbose=False)[0]


def checkpoint(name):
    model, _config = load_checkpoint(root / f"mlx-{name}")
    model.eval()
    return model


def mtp(hk=HK, expand=None):
    # The MTP target of a GGUF: the owned classes, armed as the MTP loader
    # arms them. ``expand`` is (model, layout): the weights of that model
    # with one K head per V head, paired in that order.
    config = text_raw(hk, model_type="qwen3_5", kv_head_layout="tiled")
    mx.random.seed(0)
    model, config = build_model(config, mtp=True)
    model.eval()
    mx.eval(model.parameters())
    if expand is not None:
        source, layout = expand
        idx = mx.array(qkv_rows(layout))
        weights = []
        for name, value in tree_flatten(source.parameters()):
            if name.endswith(("linear_attn.in_proj_qkv.weight",
                              "linear_attn.conv1d.weight")):
                value = value[idx]
            weights.append((name, value))
        model.load_weights(weights, strict=True)
        mx.eval(model.parameters())
    _arm_spec_target(model, config, log=lambda *a, **k: None)
    return model


def qwen3_next():
    # A Qwen3-Next model as the loader builds it: asymmetric heads in the
    # grouped order, on the copy's scans.
    mx.random.seed(0)
    model, _config = build_model(dict(
        model_type="qwen3_next", hidden_size=H, intermediate_size=FFN,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        num_key_value_heads=KV, head_dim=HD, vocab_size=len(tokens()),
        linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=DK, linear_value_head_dim=DV,
        linear_conv_kernel_dim=CONV, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, shared_expert_intermediate_size=64,
        mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
        max_position_embeddings=1024, full_attention_interval=4,
        rope_theta=10000000.0, partial_rotary_factor=0.25,
        norm_topk_prob=True, tie_word_embeddings=False,
        kv_head_layout="grouped"))
    model.eval()
    mx.eval(model.parameters())
    return model


BUILD = {"gguf": lambda: gguf("narrow"), "mlx": lambda: checkpoint("narrow"),
         "mtp": mtp, "next": qwen3_next}


def logits(model, kind):
    # The MTP target is called through its language model, the others whole.
    lm = model.language_model if kind == "mtp" else model
    cache = lm.make_cache() if kind == "mtp" else make_prompt_cache(lm)
    out = []
    for chunk in (ids[:, :5], ids[:, 5:6]):
        y = lm(chunk, cache=cache)
        y = getattr(y, "logits", y).astype(mx.float32)
        mx.eval(y)
        out.append(np.array(y).tolist())
    return {"prefill": out[0], "decode": out[1]}


ids = mx.array(np.random.default_rng(0).integers(3, len(tokens()), size=(1, 6)))
models = {kind: BUILD[kind]() for kind in order}
out = {}
for kind, model in models.items():
    out[kind] = {"model": logits(model, kind)}
if references:
    for kind, model in models.items():
        own, other = (("tiled", "grouped") if kind in ("gguf", "mtp")
                      else ("grouped", "tiled"))
        if kind == "gguf":
            pair = gguf(f"wide-{own}"), gguf(f"wide-{other}")
        elif kind == "mlx":
            pair = checkpoint(f"wide-{own}"), checkpoint(f"wide-{other}")
        elif kind == "mtp":
            pair = mtp(HV, (model, own)), mtp(HV, (model, other))
        else:
            continue
        out[kind]["reference"] = logits(pair[0], kind)
        out[kind]["other"] = logits(pair[1], kind)
print(json.dumps(out))
"""

_LOADER_CHILD = _COMMON + r"""
from pathlib import Path

root, loader = Path(sys.argv[1]), sys.argv[2]

import gmlx
import gmlx.load.config_synth as config_synth
import gmlx.load.loader as text_loader
import gmlx.load.vlm as vlm_loader
import gmlx.spec.mtp_load as mtp_loader

module = {"text": text_loader, "vlm": vlm_loader, "mtp": mtp_loader}[loader]
check = module.require_tiled_gdn
synthesize = config_synth.synthesize_config


def load(lost):
    # ``lost``: kv_head_layout leaves the config between its synthesis and
    # the model classes.
    if loader == "mtp":
        # Both MTP loaders arm the tree they built through this function.
        config = text_raw(HK, model_type="qwen3_5", kv_head_layout="tiled")
        if lost:
            del config["kv_head_layout"]
        model, config = text_loader.build_model(config, mtp=True)
        mtp_loader._arm_spec_target(model, config, log=lambda *a, **k: None)
        return

    def synthesized(*args, **kw):
        config = synthesize(*args, **kw)
        if lost:
            del config["kv_head_layout"]
        return config

    config_synth.synthesize_config = synthesized
    vlm_loader.synthesize_config = synthesized
    if loader == "text":
        gmlx.load_model(str(root / "narrow.gguf"), verbose=False)
    else:
        vlm_loader.load_vlm_model(str(root / "narrow.gguf"),
                                  str(root / "mmproj.gguf"))


class Reached(Exception):
    pass


seen = []


def stop_at_the_check(model, config):
    # The check as the loader calls it, on the tree the loader built. The
    # load ends here: the toy pair has no vision weights to install.
    seen.append({
        "layout": config.get("kv_head_layout"),
        "tiled": sorted({bool(getattr(m, "gdn_kv_tiled", False))
                         for m in model.modules()
                         if hasattr(m, "num_k_heads")})})
    check(model, config)
    raise Reached()


out = {}
module.require_tiled_gdn = stop_at_the_check
try:
    load(lost=False)
    out["intact"] = "loaded"
except Reached:
    out["intact"] = "reached the check"
module.require_tiled_gdn = check
try:
    load(lost=True)
    out["lost"] = "loaded"
except Exception as e:
    out["lost"] = f"{type(e).__name__}: {e}"
out["seen"] = seen
print(json.dumps(out))
"""

_results: dict = {}


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """The toy GGUFs and MLX checkpoints: the pair under test, and for
    each a reference with one K head per V head in either order."""
    root = tmp_path_factory.mktemp("qwen35_text_kv_layout")
    r = subprocess.run([sys.executable, "-c", _MINT, str(root)],
                       capture_output=True, text=True, timeout=900,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    return root


def _run(toy, order, device, references=False):
    key = (tuple(order), device, references)
    if key not in _results:
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, str(toy), json.dumps(list(order)),
             device, "references" if references else "alone"],
            capture_output=True, text=True, timeout=900, env=dict(os.environ))
        assert r.returncode == 0, r.stderr[-4000:]
        _results[key] = json.loads(r.stdout.strip().splitlines()[-1])
    return _results[key]


def _gap(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).max())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", _TILED + _GROUPED)
@pytest.mark.parametrize("order", _ORDERS, ids=("tiled_first", "grouped_first"))
def test_each_model_keeps_its_own_head_order(toy, order, kind, device):
    both = _run(toy, order, device, references=True)[kind]
    alone = _run(toy, (kind,), device)[kind]
    where = f"{kind} model on {device}, loaded {' then '.join(order)}"
    for phase in ("prefill", "decode"):
        got = both["model"][phase]
        same = _gap(got, alone["model"][phase])
        assert same <= ALONE_ATOL, (
            f"{where}, {phase}: {same} from the model alone in its process")
        if kind in _WITH_REFERENCE:
            ref = _gap(got, both["reference"][phase])
            assert ref < REF_ATOL, f"{where}, {phase}: {ref} from its reference"
            apart = _gap(got, both["other"][phase])
            assert apart > ORDERS_DIFFER, (
                f"{where}, {phase}: {apart} from the other order")


def _toy_config(**over):
    config = dict(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        linear_num_value_heads=6, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=1024, tie_word_embeddings=False, head_dim=32,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.25},
        full_attention_interval=4, kv_head_layout="tiled")
    config.update(over)
    return config


@pytest.mark.parametrize("mtp", (False, True), ids=("text", "mtp"))
def test_a_gguf_tree_that_would_pair_heads_grouped_is_refused(mtp):
    """A GGUF config that reaches the model classes without its
    ``kv_head_layout`` builds layers in the grouped order, which gives
    wrong text and no error. ``require_tiled_gdn`` refuses the tree, and
    ``test_each_gguf_loader_checks_the_tree_it_built`` holds each loader to
    the call."""
    from gmlx.load.loader import build_model
    from gmlx.upstream.gdn_patches import require_tiled_gdn

    model, config = build_model(_toy_config(), mtp=mtp)
    require_tiled_gdn(model, config)

    lost = _toy_config()
    del lost["kv_head_layout"]
    model, config = build_model(lost, mtp=mtp)
    with pytest.raises(RuntimeError, match="grouped order"):
        require_tiled_gdn(model, config)


def test_the_check_reads_head_counts_under_text_config():
    """A container config holds the head counts under ``text_config``."""
    from gmlx.load.loader import build_model
    from gmlx.upstream.gdn_patches import require_tiled_gdn

    model, config = build_model(_toy_config())
    require_tiled_gdn(model, {"model_type": "qwen3_5", "text_config": config})

    lost = _toy_config()
    del lost["kv_head_layout"]
    model, config = build_model(lost)
    with pytest.raises(RuntimeError, match="grouped order"):
        require_tiled_gdn(model, {"model_type": "qwen3_5",
                                  "text_config": config})


@pytest.mark.parametrize("loader", ("text", "vlm", "mtp"))
def test_each_gguf_loader_checks_the_tree_it_built(toy, loader):
    """The text loader, the VLM loader and the arming of an MTP target each
    run the check on the tree they built. With the config intact the tree
    is tiled and passes, and with ``kv_head_layout`` lost on the way to the
    model classes the load is refused."""
    r = subprocess.run(
        [sys.executable, "-c", _LOADER_CHILD, str(toy), loader],
        capture_output=True, text=True, timeout=900, env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["intact"] == "reached the check", out
    assert out["seen"] == [{"layout": "tiled", "tiled": [True]}], out
    assert out["lost"].startswith("RuntimeError: "), out
    assert "grouped order" in out["lost"], out


def _config_classes():
    from gmlx._vendor.mlx_lm.models.qwen3_5 import TextModelArgs
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig
    from gmlx._vendor.mlx_vlm.models.qwen3_5_moe.config import (
        TextConfig as MoeTextConfig)

    return {"lm_qwen3_5": TextModelArgs, "vlm_qwen3_5": TextConfig,
            "vlm_qwen3_5_moe": MoeTextConfig}


@pytest.mark.parametrize(
    "name", ("lm_qwen3_5", "vlm_qwen3_5", "vlm_qwen3_5_moe"))
def test_a_config_that_names_no_head_order_is_grouped(name):
    """The config of an MLX checkpoint names no head order, and each copied
    text config then says grouped. A config that says tiled keeps it."""
    raw = _toy_config(num_experts=4, num_experts_per_tok=2,
                      shared_expert_intermediate_size=64,
                      moe_intermediate_size=64)
    cls = _config_classes()[name]
    assert cls.from_dict(raw).kv_head_layout == "tiled"
    del raw["kv_head_layout"]
    assert cls.from_dict(raw).kv_head_layout == "grouped"


def test_a_grouped_or_symmetric_gguf_tree_passes_the_check():
    from gmlx.load.loader import build_model
    from gmlx.upstream.gdn_patches import require_tiled_gdn

    # One K head per V head: the two orders are the same pairing.
    symmetric = _toy_config(linear_num_key_heads=6)
    del symmetric["kv_head_layout"]
    model, config = build_model(symmetric)
    require_tiled_gdn(model, config)
    # A Qwen3-Next GGUF keeps the grouped order of its checkpoint.
    grouped = _toy_config(
        model_type="qwen3_next", kv_head_layout="grouped", num_experts=4,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        shared_expert_intermediate_size=64, moe_intermediate_size=64,
        rope_theta=10000000.0, partial_rotary_factor=0.25,
        norm_topk_prob=True)
    del grouped["rope_parameters"]
    model, config = build_model(grouped)
    require_tiled_gdn(model, config)
