"""The thinking-budget message through the batch engine: its ids leave the
serve criteria by the forced-token replacement, each reports the model's
raw logprob of that id, every forced id counts as a completion token, and
a message that holds EOS ends the reply."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models import llama
from gmlx._vendor.mlx_lm.models.llama import ModelArgs
from gmlx._vendor.mlx_vlm.sample_utils import make_logits_processors
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.gen.thinking_budget import seq_thinking_budget_criteria_cls
from gmlx.models.vlm_text_only import TextOnlyModel
from gmlx.serve.row_sampling import RowSettings

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

_V = 64
_K = 4
START, END, NL, EOS = 50, 51, 10, 63
MESSAGE = [20, 21, 22]
PROMPT = [3, 9, 4, START]


class _Tok:
    _MAP = {"<think>": [START], "</think>": [END], "\n": [NL],
            "Go now": MESSAGE, "Bye": [23, EOS]}

    def encode(self, text, add_special_tokens=True):
        return list(self._MAP.get(text, []))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    inner = llama.Model(ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=_V))
    m = TextOnlyModel(inner)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def _criteria(message):
    return seq_thinking_budget_criteria_cls()(
        tokenizer=_Tok(), thinking_budget=2, thinking_end_token="</think>",
        thinking_start_token="<think>", enable_thinking=True,
        prompt_open_thinking=True, message=message, reasoning_ids=PROMPT)


def _decode(model, criteria, n, *, eos=()):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria(list(eos), None)))
    gen = ar.BatchGenerator(model.language_model, proc, prefill_step_size=None,
                            max_tokens=n, logprobs_cap=_K)
    lm = model.language_model
    # The model never writes a marker or EOS itself, so only the budget
    # closes the block.
    ban = make_logits_processors({START: -1e4, END: -1e4, EOS: -1e4})
    (uid,) = gen.insert(
        [PROMPT], n, thinking_budget_criteria=[criteria],
        logits_processors=[ban],
        row_settings=[RowSettings(logprobs=True, top_k=_K)],
        prompt_kwargs=[{"inputs_embeds": lm.input_embeds(mx.array([PROMPT]))}])
    out = []
    for _ in range(4 * n):
        if not gen.has_work:
            break
        _p, resp = gen.next()
        out += [r for r in resp if r.uid == uid]
    gen.close()
    return out


def test_message_tokens_take_forced_replacement_path(model):
    crit = _criteria("Go now")
    pops = []
    pop = crit.pop_forced_token_id

    def spy():
        tid = pop()
        pops.append(tid)
        return tid

    crit.pop_forced_token_id = spy
    resps = _decode(model, crit, 8)
    toks = [r.token for r in resps]
    assert toks[2:6] == MESSAGE + [END]
    assert [t for t in pops if t is not None] == MESSAGE + [END]
    # Each forced id reports the raw f32 logprob the model gave it.
    logits = model.language_model(mx.array([PROMPT + toks[:-1]]))
    logits = getattr(logits, "logits", logits)[0].astype(mx.float32)
    ref = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    for i, r in enumerate(resps):
        assert r.token_logprob == pytest.approx(
            ref[len(PROMPT) - 1 + i][r.token], abs=3e-2)


def test_forced_tokens_count_as_completion_tokens(model):
    resps = _decode(model, _criteria("Go now"), 5)
    assert [r.token for r in resps][2:] == MESSAGE
    assert resps[-1].finish_reason == "length"


def test_message_with_eos_ends_reply(model):
    resps = _decode(model, _criteria("Bye"), 12, eos=[EOS])
    assert [r.token for r in resps][2:] == [23, EOS]
    assert resps[-1].finish_reason == "stop"
