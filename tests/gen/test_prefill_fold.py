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


def _served_gemma4(monkeypatch, shared=4):
    """A Gemma 4 E2B-shaped text model, cut to 10 layers and width 64,
    wrapped through the serve loader: per-layer inputs, KV-shared layers,
    a tied head and the final logit softcap, as the real checkpoint has.
    With no KV-shared layers it has the 31B's layer plan."""
    from gmlx.serve import bridge_vlm

    real = json.loads((_FIXTURES / "gemma4_e2b_it_config.json").read_text())
    real = real.get("text_config", real)
    cfg = dict(real, hidden_size=64, num_hidden_layers=10,
               intermediate_size=64, num_attention_heads=2, head_dim=16,
               global_head_dim=32, num_key_value_heads=1,
               num_kv_shared_layers=shared, hidden_size_per_layer_input=8,
               vocab_size=_V, vocab_size_per_layer_input=_V,
               sliding_window=4, layer_types=real["layer_types"][:10])
    assert cfg["tie_word_embeddings"] and cfg["final_logit_softcapping"]
    mx.random.seed(0)
    raw = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(cfg))
    mx.eval(raw.parameters())
    monkeypatch.setattr(bridge_vlm, "load_model",
                        lambda *a, **k: (raw, cfg, object()))
    monkeypatch.setattr(bridge_vlm, "_make_text_processor", lambda tok: None)
    model, _proc, _config = bridge_vlm.load_serveable_model("tiny.gguf")
    return model, cfg


class _LayerWidths:
    """The columns each decoder layer took and returned, per forward."""

    def __init__(self, monkeypatch):
        self.calls = []
        layer = gemma4_text.DecoderLayer.__call__

        def run(this, x, *a, **kw):
            out = layer(this, x, *a, **kw)
            self.calls.append(
                (this.layer_idx, (int(x.shape[1]), int(out[0].shape[1]))))
            return out

        monkeypatch.setattr(gemma4_text.DecoderLayer, "__call__", run)

    def forwards(self, n_layers):
        return [dict(self.calls[i:i + n_layers])
                for i in range(0, len(self.calls), n_layers)]


@pytest.mark.parametrize("shared", [4, 0])
def test_a_served_gemma4_runs_the_layers_after_its_cache_writes_on_one_column(
        monkeypatch, shared):
    model, cfg = _served_gemma4(monkeypatch, shared)
    assert model.language_model.supports_logits_to_keep
    seen = _LayerWidths(monkeypatch)
    _first_tokens(model, [list(range(1, 13))])
    n = cfg["num_hidden_layers"]
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


@pytest.mark.parametrize("shared", [4, 0])
def test_a_served_gemma4_folds_to_the_tokens_of_the_last_column_alone(
        monkeypatch, shared):
    # Prompts inside and past the sliding window and the prefill step, in
    # one left-padded batch.
    model, _cfg = _served_gemma4(monkeypatch, shared)
    prompts = [[3, 9, 4, 7, 1], [5, 2, 8], list(range(1, 13)),
               list(range(40, 60))]
    folded = _first_tokens(model, prompts)
    alone = []
    for p in prompts:
        alone += _first_tokens(model, [p])
    _unflagged(monkeypatch)
    _close(folded, _first_tokens(model, prompts))
    _close(alone, [_first_tokens(model, [p])[0] for p in prompts])
