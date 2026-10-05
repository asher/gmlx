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

from gmlx._vendor.mlx_lm.models import gemma4_text, llama
from gmlx._vendor.mlx_lm.models.llama import ModelArgs
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.models.vlm_text_only import LanguageModel, TextOnlyModel

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
        inner = gemma4_text.Model(gemma4_text.ModelArgs(
            hidden_size=32, num_hidden_layers=3, intermediate_size=64,
            num_attention_heads=2, head_dim=16, global_head_dim=32,
            num_key_value_heads=1, vocab_size=_V,
            vocab_size_per_layer_input=_V, hidden_size_per_layer_input=0,
            num_kv_shared_layers=0, sliding_window=4,
            sliding_window_pattern=3))
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

    def __init__(self, monkeypatch):
        self.trunk, self.head = [], []
        trunk, head = llama.LlamaModel.__call__, llama.Model.__call__

        def run_trunk(this, inputs, *a, **kw):
            out = trunk(this, inputs, *a, **kw)
            self.trunk.append(int(out.shape[1]))
            return out

        def run_head(this, inputs, *a, **kw):
            out = head(this, inputs, *a, **kw)
            self.head.append(int(out.shape[1]))
            return out

        monkeypatch.setattr(llama.LlamaModel, "__call__", run_trunk)
        monkeypatch.setattr(llama.Model, "__call__", run_head)


def _first_tokens(model, prompts):
    """Each prompt's first token and its log probabilities, from one batch
    of the serve engine."""
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(model.language_model, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=1, prefill_step_size=_STEP,
                            top_logprobs_k=_V)
    lm = model.language_model
    uids = gen.insert(prompts, 1, prompt_kwargs=[
        {"inputs_embeds": lm.input_embeds(mx.array([p]))} for p in prompts])
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


def _close(a, b):
    # The two paths run the trunk at other GEMM widths, and f32 GEMM runs at
    # TF32 on the GPU.
    for (tok_a, lp_a), (tok_b, lp_b) in zip(a, b, strict=True):
        assert tok_a == tok_b
        assert sorted(lp_a) == sorted(lp_b)
        np.testing.assert_allclose([lp_a[k] for k in sorted(lp_a)],
                                   [lp_b[k] for k in sorted(lp_b)],
                                   atol=2e-2)


def test_the_last_chunk_and_the_first_token_run_in_one_forward(monkeypatch):
    model = _model()
    assert model.language_model.supports_logits_to_keep
    seen = _Forwards(monkeypatch)
    _first_tokens(model, [[3, 9, 4, 7, 1]])
    # The prompt's forwards, then the decode step queued with the first
    # token.
    assert (seen.trunk, seen.head) == ([5, 1], [1, 1])
    seen.trunk.clear(), seen.head.clear()
    _first_tokens(model, [list(range(1, 13))])
    assert (seen.trunk, seen.head) == ([8, 4, 1], [8, 1, 1])


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
    assert (seen.trunk, seen.head) == ([4, 1, 1], [4, 1, 1])


def test_a_speculative_batch_keeps_the_last_column_alone():
    lm = _model().language_model
    batch = SimpleNamespace(
        model=lm, draft_model=object(), prefill_step_size=_STEP,
        prompt_cache=[], _inputs_embeds=mx.zeros((1, 5, 32)),
        _next_apc_checkpoint_column=lambda: None)
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


def _serve_wrapped(monkeypatch, raw, cfg):
    from gmlx.serve import bridge_vlm

    mx.eval(raw.parameters())
    monkeypatch.setattr(bridge_vlm, "load_model",
                        lambda *a, **k: (raw, cfg, object()))
    monkeypatch.setattr(bridge_vlm, "_make_text_processor", lambda tok: None)
    model, _proc, _config = bridge_vlm.load_serveable_model("tiny.gguf")
    return model


def _served_gemma4(monkeypatch, shared=4):
    """The copy's mlx-lm Gemma 4 text model on that config, wrapped through
    the serve loader."""
    cfg = _gemma4_config(shared)
    mx.random.seed(0)
    raw = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(cfg))
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

    def __init__(self, monkeypatch, layer_cls=gemma4_text.DecoderLayer):
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
    # A prompt chunk, the folded last chunk with the first token, and the
    # decode step queued with it. The folded forward runs its KV-writing
    # layers on the chunk, the last of them returns the last column, and
    # the KV-shared layers run that column alone.
    chunk, folded, decode = seen.forwards(n)
    assert set(chunk.values()) == {(8, 8)}
    assert {folded[i] for i in range(last_writer)} == {(4, 4)}
    assert folded[last_writer] == (4, 1)
    assert {folded[i] for i in range(last_writer + 1, n)} <= {(1, 1)}
    assert set(decode.values()) == {(1, 1)}


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
def test_a_served_gemma4_runs_the_layers_after_its_cache_writes_on_one_column(
        monkeypatch, shared):
    model, cfg = _served_gemma4(monkeypatch, shared)
    assert model.language_model.supports_logits_to_keep
    seen = _LayerWidths(monkeypatch)
    _folded_layer_widths(model, seen, cfg["num_hidden_layers"], shared)


@pytest.mark.parametrize("shared", [4, 0])
def test_a_served_gemma4_folds_to_the_tokens_of_the_last_column_alone(
        monkeypatch, shared):
    model, _cfg = _served_gemma4(monkeypatch, shared)
    _folds_to_the_last_column_alone(monkeypatch, model)


@pytest.mark.parametrize("shared", [4, 0])
def test_a_gemma4_the_loader_builds_is_the_owned_model_and_folds_on_one_column(
        monkeypatch, shared):
    from gmlx.models.gemma4 import language, text

    model, cfg = _loaded_gemma4(monkeypatch, shared)
    assert type(model.language_model._model) is text.Model
    assert model.language_model.supports_logits_to_keep
    seen = _LayerWidths(monkeypatch, language.DecoderLayer)
    _folded_layer_widths(model, seen, cfg["num_hidden_layers"], shared)


@pytest.mark.parametrize("shared", [4, 0])
def test_a_gemma4_the_loader_builds_folds_to_the_tokens_of_the_last_column_alone(
        monkeypatch, shared):
    model, _cfg = _loaded_gemma4(monkeypatch, shared)
    assert model.language_model.supports_logits_to_keep
    _folds_to_the_last_column_alone(monkeypatch, model)


def test_the_owned_gemma4_and_the_copy_fold_to_the_same_first_tokens(
        monkeypatch):
    # The same weights in both classes: the parameter trees match.
    owned, _cfg = _loaded_gemma4(monkeypatch)
    copy, _cfg = _served_gemma4(monkeypatch)
    copy.language_model._model.update(owned.language_model._model.parameters())
    prompts = [[3, 9, 4, 7, 1], list(range(1, 13)), list(range(40, 60))]
    _close(_first_tokens(owned, prompts), _first_tokens(copy, prompts))


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
    # A prompt chunk, the folded last chunk with the first token, and the
    # decode step queued with it. The last layer writes its cache from the
    # chunk and returns the last column.
    chunk, last, decode = seen.forwards(_G4_LAYERS)
    assert set(chunk.values()) == {(8, 8)}
    assert (last[0], last[1], last[2]) == ((4, 4), (4, 4), (4, 1))
    assert set(decode.values()) == {(1, 1)}
    _unflagged(monkeypatch)
    _close(folded, _first_tokens(model, [prompt]))
