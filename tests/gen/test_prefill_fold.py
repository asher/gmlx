"""The serve engine's last prefill forward. A model that keeps only the last
logits runs the last prompt chunk and the first token in one forward, as
mlx-vlm 0.6.15 did, and its head projects one column. A model that
projects every column, and a speculative batch, run the last prompt column
alone, so the head never materializes a chunk's logits."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models import llama
from gmlx._vendor.mlx_lm.models.llama import ModelArgs
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.models.gemma4 import language as gemma4_language
from gmlx.models.gemma4 import text as gemma4_text
from gmlx.models.vlm_text_only import LanguageModel, TextOnlyModel
from pristine import pristine

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

_V = 64
_STEP = 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")


def _model(arch="llama"):
    mx.random.seed(0)
    if arch == "gemma4_text":
        # Sliding layers with a window of 4, so their rotating caches wrap
        # inside the prompts, a full-attention layer and a logit softcap.
        inner = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(dict(
            model_type="gemma4_text",
            hidden_size=32, num_hidden_layers=3, intermediate_size=64,
            num_attention_heads=2, head_dim=16, global_head_dim=32,
            num_key_value_heads=1, vocab_size=_V,
            vocab_size_per_layer_input=_V, hidden_size_per_layer_input=0,
            num_kv_shared_layers=0, sliding_window=4,
            sliding_window_pattern=3)))
    else:
        inner = llama.Model(ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=_V))
    model = TextOnlyModel(inner)
    mx.eval(model.parameters())
    return model


class _Forwards:
    """The columns each forward's trunk ran, and the columns its head
    projected."""

    def __init__(self, monkeypatch, trunk_cls=llama.LlamaModel,
                 head_cls=llama.Model):
        self.trunk, self.head = [], []
        trunk, head = trunk_cls.__call__, head_cls.__call__

        def run_trunk(this, inputs, *a, **kw):
            out = trunk(this, inputs, *a, **kw)
            self.trunk.append(int(out.shape[1]))
            return out

        def run_head(this, inputs, *a, **kw):
            out = head(this, inputs, *a, **kw)
            self.head.append(int(out.shape[1]))
            return out

        monkeypatch.setattr(trunk_cls, "__call__", run_trunk)
        monkeypatch.setattr(head_cls, "__call__", run_head)


def _first_tokens(model, prompts, row_settings=None, **gen_kw):
    """Each prompt's first token and its log probabilities, from one batch
    of the serve engine."""
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(model.language_model, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=1, prefill_step_size=_STEP,
                            top_logprobs_k=_V, **gen_kw)
    lm = model.language_model
    uids = gen.insert(prompts, 1, prompt_kwargs=[
        {"inputs_embeds": lm.input_embeds(mx.array([p]))} for p in prompts],
        row_settings=row_settings)
    out = {}
    while gen.has_work:
        _p, resp = gen.next()
        for r in resp:
            out.setdefault(r.uid, (int(r.token), dict(r.top_logprobs or ())))
    gen.close()
    return [out[u] for u in uids]


def _unflagged(monkeypatch):
    monkeypatch.setattr(LanguageModel, "supports_logits_to_keep",
                        property(lambda self: False))


def _close(a, b, atol=2e-2):
    # The two paths run the trunk at other GEMM widths, and f32 GEMM runs at
    # TF32 on the GPU.
    for (tok_a, lp_a), (tok_b, lp_b) in zip(a, b, strict=True):
        assert tok_a == tok_b
        assert sorted(lp_a) == sorted(lp_b)
        np.testing.assert_allclose([lp_a[k] for k in sorted(lp_a)],
                                   [lp_b[k] for k in sorted(lp_b)],
                                   atol=atol)


def test_the_last_chunk_and_the_first_token_run_in_one_forward(monkeypatch):
    model = _model()
    assert model.language_model.supports_logits_to_keep
    seen = _Forwards(monkeypatch)
    _first_tokens(model, [[3, 9, 4, 7, 1]])
    # The prompt's forwards only: a one-token request ends on its first
    # token, so no decode step is queued.
    assert (seen.trunk, seen.head) == ([5], [1])
    seen.trunk.clear(), seen.head.clear()
    _first_tokens(model, [list(range(1, 13))])
    assert (seen.trunk, seen.head) == ([8, 4], [8, 1])


@pytest.mark.parametrize("arch", ["llama", "gemma4_text"])
def test_the_folded_first_token_matches_the_last_column_alone(monkeypatch,
                                                              arch):
    model = _model(arch)
    assert model.language_model.supports_logits_to_keep
    prompts = [[3, 9, 4, 7, 1], [5, 2, 8], list(range(1, 13))]
    folded = _first_tokens(model, prompts)
    _unflagged(monkeypatch)
    alone = _first_tokens(model, prompts)
    _close(folded, alone)


def test_a_model_that_projects_every_column_runs_the_last_column_alone(
        monkeypatch):
    model = _model()
    _unflagged(monkeypatch)
    seen = _Forwards(monkeypatch)
    _first_tokens(model, [[3, 9, 4, 7, 1]])
    assert (seen.trunk, seen.head) == ([4, 1], [4, 1])


def _short_batch(lm, n=5, draft_model=None):
    """The fields needs_processing reads, for an ``n``-token prompt."""
    return SimpleNamespace(
        model=lm, draft_model=draft_model, prefill_step_size=_STEP,
        prompt_cache=[], _inputs_embeds=mx.zeros((1, n, 32)),
        _next_apc_checkpoint_column=lambda: None)


@pytest.mark.parametrize("arch", ["llama", "qwen3_5"])
def test_a_speculative_batch_keeps_the_last_column_alone(monkeypatch, arch):
    lm = (_qwen35(monkeypatch) if arch == "qwen3_5"
          else _model()).language_model
    batch = _short_batch(lm, draft_model=object())
    assert ar.PromptProcessingBatch.needs_processing(batch) is True
    batch.draft_model = None
    assert ar.PromptProcessingBatch.needs_processing(batch) is False


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _gemma4_config(shared):
    """A Gemma 4 E2B text config, cut to 10 layers and width 64:
    per-layer inputs, KV-shared layers, a tied head and the final logit
    softcap, as the real checkpoint has. With no KV-shared layers it has
    the 31B's layer plan."""
    real = json.loads((_FIXTURES / "gemma4_e2b_it_config.json").read_text())
    real = real.get("text_config", real)
    cfg = dict(real, hidden_size=64, num_hidden_layers=10,
               intermediate_size=64, num_attention_heads=2, head_dim=16,
               global_head_dim=32, num_key_value_heads=1,
               num_kv_shared_layers=shared, hidden_size_per_layer_input=8,
               vocab_size=_V, vocab_size_per_layer_input=_V,
               sliding_window=4, layer_types=real["layer_types"][:10])
    assert cfg["tie_word_embeddings"] and cfg["final_logit_softcapping"]
    return cfg


def _serve_wrapped(monkeypatch, raw, cfg, **load):
    from gmlx.serve import bridge_vlm

    mx.eval(raw.parameters())
    monkeypatch.setattr(bridge_vlm, "load_model",
                        lambda *a, **k: (raw, cfg, object()))
    monkeypatch.setattr(bridge_vlm, "_make_text_processor", lambda tok: None)
    model, _proc, _config = bridge_vlm.load_serveable_model("tiny.gguf",
                                                            **load)
    return model


def _upstream_gemma4(monkeypatch, shared=4):
    """mlx-lm's own Gemma 4 text model on that config, from the upstream
    file, wrapped through the serve loader. The copied path holds the owned
    model."""
    upstream = pristine("gmlx._vendor.mlx_lm.models.gemma4_text")
    cfg = _gemma4_config(shared)
    mx.random.seed(0)
    raw = upstream.Model(upstream.ModelArgs.from_dict(cfg))
    return _serve_wrapped(monkeypatch, raw, cfg), cfg


def _loaded_gemma4(monkeypatch, shared=4):
    """The model the loader builds for that config, wrapped through the
    serve loader. The loader's class resolver picks the class."""
    from gmlx.load import loader

    cfg = _gemma4_config(shared)
    assert cfg["model_type"] == "gemma4_text"
    mx.random.seed(0)
    raw, _ = loader.build_model(cfg)
    return _serve_wrapped(monkeypatch, raw, cfg), cfg


class _LayerWidths:
    """The columns each decoder layer took and returned, per forward."""

    def __init__(self, monkeypatch, layer_cls=gemma4_language.DecoderLayer):
        self.calls = []
        layer = layer_cls.__call__

        def run(this, x, *a, **kw):
            out = layer(this, x, *a, **kw)
            self.calls.append(
                (this.layer_idx, (int(x.shape[1]), int(out[0].shape[1]))))
            return out

        monkeypatch.setattr(layer_cls, "__call__", run)

    def forwards(self, n_layers):
        return [dict(self.calls[i:i + n_layers])
                for i in range(0, len(self.calls), n_layers)]


def _folded_layer_widths(model, seen, n, shared):
    _first_tokens(model, [list(range(1, 13))])
    last_writer = n - shared - 1
    # A prompt chunk and the folded last chunk with the first token. The
    # folded forward runs its KV-writing layers on the chunk, the last of
    # them returns the last column, and the KV-shared layers run that
    # column alone.
    chunk, folded = seen.forwards(n)
    assert set(chunk.values()) == {(8, 8)}
    assert {folded[i] for i in range(last_writer)} == {(4, 4)}
    assert folded[last_writer] == (4, 1)
    assert {folded[i] for i in range(last_writer + 1, n)} <= {(1, 1)}


def _folds_to_the_last_column_alone(monkeypatch, model):
    # Prompts inside and past the sliding window and the prefill step, in
    # one left-padded batch.
    prompts = [[3, 9, 4, 7, 1], [5, 2, 8], list(range(1, 13)),
               list(range(40, 60))]
    folded = _first_tokens(model, prompts)
    alone = []
    for p in prompts:
        alone += _first_tokens(model, [p])
    _unflagged(monkeypatch)
    _close(folded, _first_tokens(model, prompts))
    _close(alone, [_first_tokens(model, [p])[0] for p in prompts])


@pytest.mark.parametrize("shared", [4, 0])
def test_a_gemma4_the_loader_builds_is_the_owned_model_and_folds_on_one_column(
        monkeypatch, shared):
    model, cfg = _loaded_gemma4(monkeypatch, shared)
    assert type(model.language_model._model) is gemma4_text.Model
    assert model.language_model.supports_logits_to_keep
    seen = _LayerWidths(monkeypatch)
    _folded_layer_widths(model, seen, cfg["num_hidden_layers"], shared)


@pytest.mark.parametrize("shared", [4, 0])
def test_a_gemma4_the_loader_builds_folds_to_the_tokens_of_the_last_column_alone(
        monkeypatch, shared):
    model, _cfg = _loaded_gemma4(monkeypatch, shared)
    assert model.language_model.supports_logits_to_keep
    _folds_to_the_last_column_alone(monkeypatch, model)


def test_the_owned_gemma4_folds_to_the_first_tokens_of_mlx_lms_class(
        monkeypatch):
    # The same weights in both classes: the parameter trees match. mlx-lm's
    # class has no fold and runs the last column alone.
    owned, _cfg = _loaded_gemma4(monkeypatch)
    upstream, _cfg = _upstream_gemma4(monkeypatch)
    assert type(upstream.language_model._model) is not gemma4_text.Model
    assert owned.language_model.supports_logits_to_keep
    assert not upstream.language_model.supports_logits_to_keep
    upstream.language_model._model.update(
        owned.language_model._model.parameters())
    prompts = [[3, 9, 4, 7, 1], list(range(1, 13)), list(range(40, 60))]
    _close(_first_tokens(owned, prompts), _first_tokens(upstream, prompts))


def test_the_copied_gemma4_text_path_folds_by_the_one_explicit_allow(
        monkeypatch):
    """The copied path and the owned path are one module, and its classes
    keep the owned module name, which the copy's prefix does not cover."""
    from gmlx._vendor.mlx_lm.models import gemma4_text as copy_path
    from gmlx.models import vlm_text_only

    assert copy_path is gemma4_text
    assert gemma4_text.Model.__module__ == "gmlx.models.gemma4.text"
    assert not gemma4_text.Model.__module__.startswith(
        vlm_text_only._COPY_MODELS)
    assert vlm_text_only._OWNED_MODELS == {"gmlx.models.gemma4.text"}
    model, _cfg = _loaded_gemma4(monkeypatch)
    assert model.language_model.supports_logits_to_keep
    monkeypatch.setattr(vlm_text_only, "_OWNED_MODELS", frozenset())
    assert not model.language_model.supports_logits_to_keep


class _Cache:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


@pytest.mark.parametrize("shared", [4, 0])
def test_the_owned_gemma4_trims_at_the_last_writer_on_a_plain_cache_only(
        monkeypatch, shared):
    model, cfg = _loaded_gemma4(monkeypatch, shared)
    trunk = model.language_model._model.model
    n = cfg["num_hidden_layers"]
    first_shared = n - shared
    h = mx.zeros((1, 4, cfg["hidden_size"]))

    def kept_from(last, keep=1):
        cache = [_Cache() for _ in range(first_shared - 1)] + [last]
        return trunk._kept_from(h, cache + [None] * shared, keep)

    assert kept_from(_Cache()) == first_shared - 1
    # A quantized, TurboQuant or KVarN cache holds no plain keys, so the
    # trim waits for the first KV-shared layer.
    after = first_shared if shared else None
    assert kept_from(_Cache(bits=8)) == after
    assert kept_from(_Cache(kv_quant_scheme="turbo")) == after
    assert kept_from(_Cache(kv_quant_scheme="kvarn")) == after
    assert kept_from(_Cache(), keep=4) is None
    assert kept_from(_Cache(), keep=0) is None


_G4_HID, _G4_LAYERS, _G4_HEADS, _G4_KV, _G4_FFN = 64, 3, 2, 1, 64
_G4_HD_SLIDING, _G4_HD_FULL = 16, 32
# Two sliding layers, then a full-attention layer.
_G4_SLIDING = [True, True, False]


def _mint_tiny_gemma4(path: str) -> int:
    """A loadable gemma4 GGUF with the 31B's layer plan: sliding and full
    attention layers, no KV-shared layers, a tied head and the final logit
    softcap. Returns the vocabulary size."""
    from gguf import GGUFWriter
    from tokenizers import pre_tokenizers

    tokens = (["<s>", "</s>", "<pad>"]
              + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"])
    w = GGUFWriter(path, "gemma4")
    for key, value in (("embedding_length", _G4_HID),
                       ("block_count", _G4_LAYERS),
                       ("attention.head_count", _G4_HEADS),
                       ("attention.head_count_kv", _G4_KV),
                       ("feed_forward_length", _G4_FFN),
                       ("context_length", 1024),
                       ("attention.key_length", _G4_HD_FULL),
                       ("attention.value_length", _G4_HD_FULL),
                       ("attention.key_length_swa", _G4_HD_SLIDING),
                       ("attention.value_length_swa", _G4_HD_SLIDING),
                       ("attention.shared_kv_layers", 0),
                       ("embedding_length_per_layer_input", 0),
                       ("attention.sliding_window", 4)):
        w.add_uint32(f"gemma4.{key}", value)
    w.add_array("gemma4.attention.sliding_window_pattern", _G4_SLIDING)
    w.add_float32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("gemma4.rope.freq_base", 1e6)
    w.add_float32("gemma4.rope.freq_base_swa", 1e4)
    w.add_float32("gemma4.final_logit_softcapping", 30.0)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.merges", ["H e", "w o"])
    w.add_array("tokenizer.ggml.token_type", [3, 3, 3] + [1] * (len(tokens) - 3))
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    rng = np.random.default_rng(0)

    def t(name, *shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)

    t("token_embd.weight", len(tokens), _G4_HID)
    t("output_norm.weight", _G4_HID)
    for i, sliding in enumerate(_G4_SLIDING):
        hd = _G4_HD_SLIDING if sliding else _G4_HD_FULL
        for norm in ("attn_norm", "post_attention_norm", "ffn_norm",
                     "post_ffw_norm"):
            t(f"blk.{i}.{norm}.weight", _G4_HID)
        t(f"blk.{i}.attn_q_norm.weight", hd)
        t(f"blk.{i}.attn_k_norm.weight", hd)
        t(f"blk.{i}.attn_q.weight", _G4_HEADS * hd, _G4_HID)
        t(f"blk.{i}.attn_k.weight", _G4_KV * hd, _G4_HID)
        t(f"blk.{i}.attn_v.weight", _G4_KV * hd, _G4_HID)
        t(f"blk.{i}.attn_output.weight", _G4_HID, _G4_HEADS * hd)
        t(f"blk.{i}.ffn_gate.weight", _G4_FFN, _G4_HID)
        t(f"blk.{i}.ffn_up.weight", _G4_FFN, _G4_HID)
        t(f"blk.{i}.ffn_down.weight", _G4_HID, _G4_FFN)
        w.add_tensor(f"blk.{i}.layer_output_scale.weight",
                     np.ones((1,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return len(tokens)


def test_a_gemma4_gguf_served_as_text_builds_the_owned_model_and_folds(
        tmp_path, monkeypatch):
    # The whole serve load of a GGUF file, with nothing patched but the
    # layer call that records the widths.
    from gmlx.models.gemma4 import language, text
    from gmlx.serve import bridge_vlm

    gguf = str(tmp_path / "tiny-gemma4.gguf")
    vocab = _mint_tiny_gemma4(gguf)
    monkeypatch.setitem(globals(), "_V", vocab)
    model, _proc, _config = bridge_vlm.load_serveable_model(gguf)
    inner = model.language_model._model
    assert type(inner) is text.Model
    assert model.language_model.supports_logits_to_keep
    prompt = list(range(3, 15))
    # The load may swap in a fused subclass of the owned layer.
    (layer_cls,) = {type(layer) for layer in inner.model.layers}
    assert issubclass(layer_cls, language.DecoderLayer)
    seen = _LayerWidths(monkeypatch, layer_cls)
    folded = _first_tokens(model, [prompt])
    # A prompt chunk and the folded last chunk with the first token. The
    # last layer writes its cache from the chunk and returns the last
    # column.
    chunk, last = seen.forwards(_G4_LAYERS)
    assert set(chunk.values()) == {(8, 8)}
    assert (last[0], last[1], last[2]) == ((4, 4), (4, 4), (4, 1))
    _unflagged(monkeypatch)
    _close(folded, _first_tokens(model, [prompt]))


# Qwen3.5 dense keeps its trunk at language_model.model, one link below the
# model the text loader builds for a qwen35 GGUF. Layers 0 to 2 are
# gated-delta layers and layer 3 is full attention.
_Q35 = dict(model_type="qwen3_5", hidden_size=64, intermediate_size=128,
            linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=32, linear_value_head_dim=32,
            linear_conv_kernel_dim=4, num_hidden_layers=4,
            num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=_V,
            num_key_value_heads=2, max_position_embeddings=2048,
            tie_word_embeddings=True, head_dim=32,
            rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                             "rope_theta": 100000,
                             "partial_rotary_factor": 0.25},
            full_attention_interval=4, kv_head_layout="tiled")


def _qwen35(monkeypatch, tied=True, dtype=mx.float32, **load):
    """A Qwen3.5 dense model of the class the loader builds, served as
    text through the serve loader."""
    from gmlx._vendor.mlx_lm.models import qwen3_5

    cfg = dict(_Q35, tie_word_embeddings=tied)
    mx.random.seed(5)
    raw = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(cfg)))
    raw.set_dtype(dtype)
    return _serve_wrapped(monkeypatch, raw, cfg, **load)


def _q35_forwards(monkeypatch):
    from gmlx._vendor.mlx_lm.models import qwen3_5

    return _Forwards(monkeypatch, qwen3_5.Qwen3_5TextModel, qwen3_5.TextModel)


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_a_qwen3_5_dense_runs_a_short_prompt_in_one_forward(monkeypatch,
                                                            tied):
    from gmlx._vendor.mlx_lm.models import qwen3_5

    model = _qwen35(monkeypatch, tied)
    lm = model.language_model
    assert type(lm._model) is qwen3_5.Model
    assert hasattr(lm._model.language_model, "lm_head") is not tied
    assert lm.supports_logits_to_keep
    assert ar.PromptProcessingBatch.needs_processing(_short_batch(lm)) is False
    # The forward hands the head one column, and the text model's own trunk
    # is back after it.
    text = lm._model.language_model
    trunk = text.model
    ids = mx.array([[3, 9, 4, 7, 1]])
    for kw in ({}, {"inputs_embeds": lm.input_embeds(ids)}):
        kept = lm(ids, cache=lm.make_cache(), logits_to_keep=1, **kw).logits
        assert kept.shape == (1, 1, _V)
        assert "model" not in vars(text) and text.model is trunk
    seen = _q35_forwards(monkeypatch)
    _first_tokens(model, [[3, 9, 4, 7, 1]])
    assert (seen.trunk, seen.head) == ([5], [1])
    seen.trunk.clear(), seen.head.clear()
    _first_tokens(model, [list(range(1, 13))])
    assert (seen.trunk, seen.head) == ([8, 4], [8, 1])


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_a_qwen3_5_dense_one_forward_matches_the_split_prefill(monkeypatch,
                                                               tied):
    # The one forward runs the gated-delta scan over other chunk widths, so
    # the first tokens match within tolerance, not bit for bit.
    model = _qwen35(monkeypatch, tied)
    prompts = [[3, 9, 4, 7, 1], [5, 2, 8], list(range(1, 13))]
    folded = _first_tokens(model, prompts)
    _unflagged(monkeypatch)
    _close(folded, _first_tokens(model, prompts))


def test_qwen3_5_moe_folds_on_its_own_trunk_as_before(monkeypatch):
    """The loader builds qwen3_5_moe as the unwrapped text model, whose
    trunk is at ``model``, and it folds as it did."""
    from gmlx._vendor.mlx_lm.models import qwen3_5
    from gmlx.load import loader

    cfg = dict(_Q35, model_type="qwen3_5_moe", num_experts=4,
               num_experts_per_tok=2, moe_intermediate_size=32,
               shared_expert_intermediate_size=32)
    mx.random.seed(5)
    raw, cfg = loader.build_model(cfg)
    assert type(raw) is qwen3_5.TextModel
    model = _serve_wrapped(monkeypatch, raw, cfg)
    lm = model.language_model
    assert lm.supports_logits_to_keep
    seen = _q35_forwards(monkeypatch)
    _first_tokens(model, [[3, 9, 4, 7, 1]])
    assert (seen.trunk, seen.head) == ([5], [1])


def test_a_model_off_the_copy_below_a_copied_wrapper_projects_every_column():
    """Each model on the way to the trunk must be one whose forward runs
    only per-column steps around it."""
    import mlx.nn as nn

    from gmlx._vendor.mlx_lm.models import qwen3_5
    from gmlx.models.vlm_text_only import trunk_owner

    class _Foreign(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.model = trunk

    outer = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(_Q35)))
    lm = LanguageModel(outer)
    assert lm.supports_logits_to_keep
    outer.language_model = _Foreign(outer.language_model.model)
    assert trunk_owner(outer) is outer.language_model
    assert not lm.supports_logits_to_keep


def test_a_shadowed_trunk_is_no_owner():
    """A forward that already shadows the trunk leaves nothing to keep."""
    from gmlx._vendor.mlx_lm.models import qwen3_5
    from gmlx.models.vlm_text_only import trunk_owner

    outer = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(_Q35)))
    text = outer.language_model
    assert trunk_owner(outer) is text
    vars(text)["model"] = object()
    try:
        assert trunk_owner(outer) is None
        assert not LanguageModel(outer).supports_logits_to_keep
    finally:
        vars(text).pop("model")


def _reserving_manager(monkeypatch):
    """An autosized manager under APC_MEMORY_RESERVE_GB holding 32 idle
    blocks, whose plan has learned 64 KB per token, on a box with no
    headroom left."""
    from gmlx._vendor.mlx_vlm import apc
    from gmlx.cache.apc_manager import GmlxAPCManager
    import gmlx.gen.prefill_decay as prefill_decay
    import gmlx.serve.capacity as capacity

    gb = 1 << 30
    for k in ("APC_NUM_BLOCKS", "APC_EXACT_CACHE_ENTRIES",
              "APC_MEMORY_MAX_GB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APC_MEMORY_RESERVE_GB", "0")
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 * gb)
    monkeypatch.setattr(prefill_decay, "untracked_weight_bytes", lambda: 0.0)
    monkeypatch.setattr(prefill_decay, "streamed_tracked_bytes", lambda: 0.0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    m.autosize(SimpleNamespace(config=SimpleNamespace(
        num_hidden_layers=10, num_attention_heads=8, num_key_value_heads=4,
        head_dim=64)))
    k = mx.zeros((1, 2, 512, 16), mx.float16)
    m.release(m.store_kv_blocks(list(range(512)), [k, k], [k, k]))
    assert len(m.hash_table) == 32
    kk = mx.zeros((1, 8, 1024, 128), mx.float16)
    m.memory_plan.observe_kv([kk] * 16, [kk] * 16, live_bytes=0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 100 * gb)
    monkeypatch.setattr(apc, "_free_ram_bytes", lambda: 0)
    return m


def test_a_one_forward_prefill_evicts_as_the_constructor_sizes_require(
        monkeypatch):
    """The one forward skips the per-chunk observe. The prompt batch's
    constructor still sizes the prefill from the plan, and the reserve
    evicts the idle blocks."""
    from gmlx._vendor.mlx_vlm import apc_coordinator

    model = _qwen35(monkeypatch)
    lm = model.language_model
    m = _reserving_manager(monkeypatch)
    observed, reserves = [], []
    monkeypatch.setattr(apc_coordinator.APCCoordinator, "observe_cache",
                        lambda self, *a, **k: observed.append(a))
    prepare = type(m).prepare_prefill
    monkeypatch.setattr(m, "prepare_prefill",
                        lambda r: (reserves.append(r), prepare(m, r))[1])
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(lm, proc, sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=1, prefill_step_size=_STEP,
                            apc_manager=m)
    prompt = [3, 9, 4, 7, 1]
    seen = _q35_forwards(monkeypatch)
    gen.insert([prompt], 1, prompt_kwargs=[
        {"inputs_embeds": lm.input_embeds(mx.array([prompt]))}])
    while gen.has_work:
        gen.next()
    gen.close()
    assert seen.trunk[0] == len(prompt) and observed == []
    assert reserves and reserves[0] > 0
    assert m.stats.memory_evictions == 32
    assert len(m.hash_table) == 0


# Per dtype, how far the one-forward first-token log probabilities may sit
# from the split prefill's. f32 runs at TF32 on the GPU. bf16 allows two
# rounding steps at the size of these log probabilities (4 to 8).
_LORA_ATOL = {mx.float32: 2e-2, mx.bfloat16: 6.25e-2}


@pytest.fixture
def _rows_channel():
    from gmlx import lora_rows

    lora_rows.install_row_channel()
    yield lora_rows
    lora_rows.configure("static", 1)


def _row(scale, salt):
    from gmlx.serve.lora_request import RowLora
    from gmlx.serve.row_sampling import RowSettings

    return RowSettings(logprobs=True, top_k=_V, lora=RowLora(
        (scale,), np.array([scale], np.float32), salt, "m"))


def _mint_q35_adapter(path):
    """A LoRA adapter on a gated-delta projection, the full-attention
    query and an MLP projection."""
    from helpers import mint_lora_gguf

    rng = np.random.default_rng(0)
    d = _Q35["hidden_size"]
    shapes = {"blk.0.attn_qkv.weight": 256, "blk.3.attn_q.weight": 256,
              "blk.1.ffn_up.weight": 128}
    mint_lora_gguf(path, "qwen35", {
        name: (rng.standard_normal((2, d)) * 0.3,
               rng.standard_normal((out, 2)) * 0.3)
        for name, out in shapes.items()})
    return str(path)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16],
                         ids=["f32", "bf16"])
def test_an_adapted_qwen3_5_one_forward_matches_the_split_prefill(
        monkeypatch, tmp_path, _rows_channel, dtype):
    """Two rows at other adapter scales, in one batch, in rows mode."""
    import gmlx.load.modules as modules

    adapter = _mint_q35_adapter(tmp_path / "adapter.gguf")
    model = _qwen35(monkeypatch, dtype=dtype, adapter_gguf=adapter)
    lm = model.language_model
    assert _rows_channel.mode() == "rows"
    assert sum(isinstance(m, modules.LoRAKQuantLinear)
               for _, m in lm._model.named_modules()) == 3
    assert lm.supports_logits_to_keep
    # Both rows in one prompt batch, whatever the installed engine sets.
    pair = dict(prefill_batch_size=2, completion_batch_size=8)
    prompts = [list(range(1, 13)), list(range(30, 42))]
    rows = [_row(0.5, 11), _row(1.5, 22)]
    seen = _q35_forwards(monkeypatch)
    folded = _first_tokens(model, prompts, rows, **pair)
    # A chunk, then the last chunk in one forward that keeps one column.
    assert (seen.trunk, seen.head) == ([8, 4], [8, 1])
    bare = _first_tokens(model, prompts, [_row(0.0, 0), _row(0.0, 0)], **pair)
    _unflagged(monkeypatch)
    split = _first_tokens(model, prompts, rows, **pair)
    atol = _LORA_ATOL[dtype]
    _close(folded, split, atol=atol)
    # The adapter moves each row by more than the tolerance.
    for (_t, lp_a), (_u, lp_b) in zip(folded, bare):
        assert max(abs(lp_a[k] - lp_b[k]) for k in lp_a) > 5 * atol
