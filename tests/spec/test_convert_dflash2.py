"""scripts/convert_dflash2.py on a tiny DFlash2 checkpoint: the header keys,
the per-tensor type policy, quant bytes against gguf-py, and a load through
gmlx's dflash loader against a Ripple target."""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gguf
from gguf import GGMLQuantizationType as Q

_SCRIPT = Path(__file__).parents[2] / "scripts" / "convert_dflash2.py"

HIDDEN, HEADS, KV, HD, FFN = 256, 4, 2, 64, 512
N_LAYERS, RANK, VOCAB, KERNEL, GROUP = 5, 256, 512, 4, 16
TARGETS = [0, 2]
TARGET_LAYERS = 4
MASK = 7
CONTEXT = 64


@pytest.fixture(scope="module")
def conv():
    spec = importlib.util.spec_from_file_location("_convert_dflash2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hf_shapes():
    shapes = {
        "fc.weight": [HIDDEN, HIDDEN * len(TARGETS)],
        "hidden_norm.weight": [HIDDEN],
        "norm.weight": [HIDDEN],
        "candidate_selector.hidden_projection.weight": [RANK, HIDDEN],
        "candidate_selector.predecessor_codebook": [VOCAB, RANK],
        "candidate_selector.successor_codebook": [VOCAB, RANK],
    }
    conv_out = HIDDEN // GROUP * 2 * KERNEL
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": [HIDDEN],
            p + "post_attention_layernorm.weight": [HIDDEN],
            p + "self_attn.q_proj.weight": [HEADS * HD, HIDDEN],
            p + "self_attn.k_proj.weight": [KV * HD, HIDDEN],
            p + "self_attn.v_proj.weight": [KV * HD, HIDDEN],
            p + "self_attn.o_proj.weight": [HIDDEN, HEADS * HD],
            p + "self_attn.q_norm.weight": [HD],
            p + "self_attn.k_norm.weight": [HD],
            p + "mlp.gate_proj.weight": [FFN, HIDDEN],
            p + "mlp.up_proj.weight": [FFN, HIDDEN],
            p + "mlp.down_proj.weight": [HIDDEN, FFN],
            p + "attention_conv.base_kernel": [2, KERNEL, HIDDEN],
            p + "attention_conv.kernel_projection.weight": [conv_out, HIDDEN],
            p + "mlp_conv.base_kernel": [2, KERNEL, HIDDEN],
            p + "mlp_conv.kernel_projection.weight": [conv_out, HIDDEN],
        })
    return shapes


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = _bf16_bits(arr).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def _config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": HIDDEN, "intermediate_size": FFN,
        "num_hidden_layers": N_LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HD, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096, "vocab_size": VOCAB,
        "is_causal": False, "use_sliding_window": True, "sliding_window": 32,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "no_rope_layers": [4],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "dflash_config": {
            "block_size": 16, "conv_group_size": GROUP,
            "conv_kernel_size": KERNEL, "mask_token_id": MASK,
            "selector_rank": RANK, "selector_top_k": 8,
            "target_layer_ids": TARGETS,
        },
    }


def _write_target(path: Path) -> None:
    w = gguf.GGUFWriter(path, "ricursive")
    w.add_name("Tiny-Target")
    w.add_license("apache-2.0")
    w.add_uint32("ricursive.block_count", TARGET_LAYERS)
    w.add_uint32("ricursive.embedding_length", HIDDEN)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("dbrx")
    w.add_token_list([f"t{i}" for i in range(VOCAB)])
    w.add_token_types([1] * VOCAB)
    w.add_token_merges(["t 0"])
    w.add_eos_token_id(3)
    w.add_tensor("dummy", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("dflash2")
    ckpt = root / "ckpt"
    ckpt.mkdir()
    rng = np.random.default_rng(0)
    tensors = {name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
               for name, shape in _hf_shapes().items()}
    _write_safetensors(ckpt / "model.safetensors", tensors)
    (ckpt / "config.json").write_text(json.dumps(_config()))
    (ckpt / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: someorg/Tiny-Target\n---\n# x\n")
    target = root / "target.gguf"
    _write_target(target)
    return root, ckpt, target, tensors


def _convert(conv, pair, outtype, *extra):
    root, ckpt, target, _ = pair
    out = root / f"draft-{outtype}.gguf"
    if not out.exists():
        assert conv.main(["--checkpoint", str(ckpt), "--target-gguf", str(target),
                          "--outtype", outtype, "--out", str(out),
                          "--context-window", str(CONTEXT), *extra]) == 0
    return out


def _fields(path):
    r = gguf.GGUFReader(path)
    return r, {k: f.contents() for k, f in r.fields.items()
               if not k.startswith("GGUF.")}


# The dflash.* keys llama.cpp's converter writes for Ripple's DFlash2
# checkpoint (convert_hf_to_gguf.py at e8df20d23), plus gmlx's two.
_LLAMACPP_DFLASH_KEYS = {
    "dflash.attention.causal", "dflash.attention.head_count",
    "dflash.attention.head_count_kv", "dflash.attention.key_length",
    "dflash.attention.layer_norm_rms_epsilon",
    "dflash.attention.sliding_window", "dflash.attention.sliding_window_pattern",
    "dflash.attention.value_length", "dflash.block_count", "dflash.block_size",
    "dflash.context_length", "dflash.conv_group_size",
    "dflash.conv_kernel_size", "dflash.embedding_length",
    "dflash.feed_forward_length", "dflash.rope.freq_base",
    "dflash.selector_rank", "dflash.selector_top_k", "dflash.target_layers",
}
_GMLX_DFLASH_KEYS = {"dflash.attention.rope_pattern",
                     "dflash.attention.context_window"}


def test_header_keys(conv, pair):
    r, kv = _fields(_convert(conv, pair, "q8_0"))
    assert {k for k in kv if k.startswith("dflash.")} == (
        _LLAMACPP_DFLASH_KEYS | _GMLX_DFLASH_KEYS)
    assert kv["general.base_model.0.repo_url"] == (
        "https://huggingface.co/someorg/Tiny-Target")
    assert kv["general.architecture"] == "dflash"
    assert kv["general.base_model.0.name"] == "Tiny-Target"
    assert kv["dflash.target_layers"] == [i + 1 for i in TARGETS]
    assert kv["dflash.attention.sliding_window"] == 32
    assert kv["dflash.attention.sliding_window_pattern"] == [True] * 4 + [False]
    assert kv["dflash.attention.rope_pattern"] == [True] * 4 + [False]
    assert kv["dflash.attention.context_window"] == CONTEXT
    assert kv["dflash.attention.causal"] is False
    assert kv["dflash.selector_top_k"] == 8
    assert kv["tokenizer.ggml.mask_token_id"] == MASK
    assert kv["tokenizer.ggml.tokens"] == [f"t{i}" for i in range(VOCAB)]
    assert kv["general.file_type"] == int(gguf.LlamaFileType.MOSTLY_Q8_0)
    assert len(r.tensors) == 6 + 15 * N_LAYERS


def test_unknown_target_needs_a_context_window(conv, pair, tmp_path):
    _, ckpt, target, _ = pair
    with pytest.raises(SystemExit, match="no known context window"):
        conv.main(["--checkpoint", str(ckpt), "--target-gguf", str(target),
                   "--out", str(tmp_path / "x.gguf")])


def test_a_mismatched_target_is_refused(conv, pair, tmp_path):
    _, ckpt, _, _ = pair
    bad = tmp_path / "bad.gguf"
    w = gguf.GGUFWriter(bad, "ricursive")
    w.add_name("Tiny-Target")
    w.add_uint32("ricursive.block_count", TARGET_LAYERS)
    w.add_uint32("ricursive.embedding_length", HIDDEN * 2)
    w.add_token_list(["a"] * VOCAB)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(SystemExit, match="target embedding_length"):
        conv.main(["--checkpoint", str(ckpt), "--target-gguf", str(bad),
                   "--context-window", "8", "--out", str(tmp_path / "x.gguf")])


@pytest.mark.parametrize("outtype,qtype", [("q8_0", Q.Q8_0), ("q4_0", Q.Q4_0)])
def test_quant_bytes_equal_gguf_py(conv, pair, outtype, qtype):
    _, _, _, tensors = pair
    r, _ = _fields(_convert(conv, pair, outtype))
    by_name = {t.name: t for t in r.tensors}
    t = by_name["blk.1.ffn_down.weight"]
    assert t.tensor_type == qtype
    src = tensors["layers.1.mlp.down_proj.weight"]
    widened = (_bf16_bits(src).astype(np.uint32) << 16).view(np.float32)
    want = gguf.quants.quantize(widened, qtype)
    assert np.array_equal(np.asarray(t.data).reshape(want.shape), want)


def test_q4_k_m_follows_use_more_bits(conv, pair):
    r, kv = _fields(_convert(conv, pair, "q4_k_m"))
    types = {t.name: t.tensor_type for t in r.tensors}
    for i in range(N_LAYERS):
        more = Q.Q6_K if i in (2, 4) else Q.Q4_K
        assert types[f"blk.{i}.attn_v.weight"] == more, i
        assert types[f"blk.{i}.ffn_down.weight"] == more, i
        assert types[f"blk.{i}.attn_q.weight"] == Q.Q4_K
        assert types[f"blk.{i}.attn_conv_base"] == Q.F32
        assert types[f"blk.{i}.attn_norm.weight"] == Q.F32
    for name in ("fc.weight", "selector_hidden.weight",
                 "selector_predecessor.weight", "selector_successor.weight"):
        assert types[name] == Q.Q4_K, name
    assert types["enc.output_norm.weight"] == Q.F32
    assert kv["general.file_type"] == int(gguf.LlamaFileType.MOSTLY_Q4_K_M)


def test_bf16_keeps_the_source_bits(conv, pair):
    _, _, _, tensors = pair
    r, _ = _fields(_convert(conv, pair, "bf16"))
    t = {t.name: t for t in r.tensors}["fc.weight"]
    assert t.tensor_type == Q.BF16
    got = np.asarray(t.data).view(np.uint16).reshape(-1)
    assert np.array_equal(got, _bf16_bits(tensors["fc.weight"]).reshape(-1))


def _target_model():
    from gmlx.models.ricursive.model import ModelArgs
    from gmlx.models.ricursive.mtp import RicursiveSpecLM

    args = ModelArgs(
        model_type="ricursive", hidden_size=HIDDEN,
        num_hidden_layers=TARGET_LAYERS, intermediate_size=FFN,
        num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD,
        vocab_size=VOCAB, sliding_window=8,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        embedding_scale=4.0, use_valres=True)
    mx.random.seed(0)
    lm = RicursiveSpecLM(args)
    mx.eval(lm.parameters())
    return lm, {"model_type": "ricursive", "num_hidden_layers": TARGET_LAYERS,
                "hidden_size": HIDDEN, "vocab_size": VOCAB,
                "max_position_embeddings": 4096}


@pytest.mark.parametrize("outtype", ["q8_0", "q4_k_m"])
def test_the_file_loads_through_the_dflash_loader(conv, pair, outtype):
    import gmlx.spec.mtp_load as mtp_load
    from gmlx.spec.dflash_drafter import DFlash2Drafter

    path = _convert(conv, pair, outtype)
    lm, target_dict = _target_model()
    drafter = mtp_load._load_dflash_drafter(
        str(path), lm, target_dict, zero_copy=False, log=lambda *a, **k: None)
    assert isinstance(drafter, DFlash2Drafter)
    assert lm._dflash_capture == tuple(TARGETS)
    cfg = drafter.config
    assert cfg.rope_layers == [True] * 4 + [False]
    assert cfg.context_window == CONTEXT
    assert [layer.self_attn.use_rope for layer in drafter.layers] == cfg.rope_layers
    drafter.reset(lm)
    hid, _ = lm.speculative_verify_hidden(mx.array([[1, 2, 3, 4, 5]]),
                                          lm.make_cache())
    drafter.prefill_from_target_hidden(None, hid, None, None)
    drafts = drafter.draft_block(3, None, None, 8, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, 7)
