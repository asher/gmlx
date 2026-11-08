"""The thinking-budget message through the batch engine: its ids leave the
serve criteria by the forced-token replacement, each reports the model's
raw logprob of that id, every forced id counts as a completion token, and
a message that holds EOS ends the reply. A budget the prompt's reasoning
already spends forces from the first generated token, in the batch engine
and in the single-stream generate_step."""

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
# Two reasoning tokens after the start marker: a seed of 2.
SPENT = [3, 9, 4, START, 5, 6]


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


def _criteria(message, *, prompt=PROMPT, budget=2):
    return seq_thinking_budget_criteria_cls()(
        tokenizer=_Tok(), thinking_budget=budget, thinking_end_token="</think>",
        thinking_start_token="<think>", enable_thinking=True,
        prompt_open_thinking=True, message=message, reasoning_ids=prompt)


def _ban():
    # The model never writes a marker or EOS itself, so only the budget
    # closes the block.
    return make_logits_processors({START: -1e4, END: -1e4, EOS: -1e4})


def _decode_rows(model, prompts, criteria, n, *, eos=()):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria(list(eos), None)))
    gen = ar.BatchGenerator(model.language_model, proc, prefill_step_size=None,
                            max_tokens=n, logprobs_cap=_K)
    lm = model.language_model
    uids = gen.insert(
        prompts, n, thinking_budget_criteria=list(criteria),
        logits_processors=[_ban() for _ in prompts],
        row_settings=[RowSettings(logprobs=True, top_k=_K) for _ in prompts],
        prompt_kwargs=[{"inputs_embeds": lm.input_embeds(mx.array([p]))}
                       for p in prompts])
    out = {uid: [] for uid in uids}
    for _ in range(4 * n):
        if not gen.has_work:
            break
        _p, resp = gen.next()
        for r in resp:
            out[r.uid].append(r)
    gen.close()
    return [out[uid] for uid in uids]


def _decode(model, criteria, n, *, eos=(), prompt=PROMPT):
    return _decode_rows(model, [prompt], [criteria], n, eos=eos)[0]


def _raw_logprobs(model, ids):
    """The f32 log-softmax of the model's logits at each position of ids."""
    logits = model.language_model(mx.array([ids]))
    logits = getattr(logits, "logits", logits)[0].astype(mx.float32)
    return np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))


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
    ref = _raw_logprobs(model, PROMPT + toks[:-1])
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


def test_a_spent_budget_forces_the_message_first(model):
    crit = _criteria("Go now", prompt=SPENT)
    resps = _decode(model, crit, 6, prompt=SPENT)
    assert [r.token for r in resps][:4] == MESSAGE + [END]
    assert crit.thinking_token_count == 2


def test_budget_zero_at_the_start_marker_forces_the_bare_close(model):
    resps = _decode(model, _criteria(None, budget=0), 4)
    assert [r.token for r in resps][:2] == [NL, END]


def test_a_forced_first_token_reports_its_raw_logprob(model):
    first = _decode(model, _criteria("Go now", prompt=SPENT), 1,
                    prompt=SPENT)[0]
    assert first.token == MESSAGE[0]
    ref = _raw_logprobs(model, SPENT)[-1]
    assert first.token_logprob == pytest.approx(ref[MESSAGE[0]], abs=3e-2)
    # The top-k is the model's distribution at that position.
    want = np.argsort(-ref)[:_K].tolist()
    assert sorted(t for t, _ in first.top_logprobs) == sorted(want)


def test_a_row_under_its_budget_keeps_the_model_first_token(model):
    """In a two-row batch only the row whose seed meets its budget forces."""
    plain = _decode_rows(model, [SPENT, PROMPT],
                         [None, _criteria("Go now")], 1)
    forced = _decode_rows(model, [SPENT, PROMPT],
                          [_criteria("Go now", prompt=SPENT),
                           _criteria("Go now")], 1)
    assert forced[0][0].token == MESSAGE[0]
    assert forced[1][0].token == plain[1][0].token
    assert forced[1][0].token not in MESSAGE + [NL, END]


def _stream_criteria(prompt, budget, *, enable_thinking=True, message=None):
    """The criteria as the single-stream dispatch builds it."""
    from gmlx.gen.thinking_budget import marker_ids, prompt_opens_block

    tok = _Tok()
    pre = prompt_opens_block(prompt, marker_ids(tok, "<think>"),
                             marker_ids(tok, "</think>"))
    return seq_thinking_budget_criteria_cls()(
        tokenizer=tok, thinking_budget=budget, thinking_end_token="</think>",
        thinking_start_token="<think>", enable_thinking=enable_thinking,
        prompt_preopens_thinking=pre,
        prompt_open_thinking=enable_thinking and pre,
        prompt_ids=prompt[-16:], reasoning_ids=prompt, message=message)


def _stream(model, prompt, criteria, n, processors=None):
    """Drive the real generate_step, calling the criteria after each yield
    as the dispatch loop does."""
    gen = ar.generate_step(
        mx.array([prompt]), model, None, None, max_tokens=n,
        logits_processors=processors, thinking_budget_criteria=criteria)
    out = []
    for token, logprobs in gen:
        out.append((int(token), logprobs))
        if criteria is not None:
            criteria(token)
    return out


def test_single_stream_budget_zero_forces_the_first_token(model):
    out = _stream(model, PROMPT, _stream_criteria(PROMPT, 0), 3)
    assert [t for t, _ in out][:2] == [NL, END]
    ref = _raw_logprobs(model, PROMPT)[-1]
    assert float(out[0][1][NL]) == pytest.approx(ref[NL], abs=3e-2)


def test_single_stream_spent_budget_forces_the_message(model):
    crit = _stream_criteria(SPENT, 2, message="Go now")
    out = _stream(model, SPENT, crit, 5, processors=_ban())
    assert [t for t, _ in out][:4] == MESSAGE + [END]


def test_single_stream_thinking_off_keeps_the_model_first_token(model):
    plain = _stream(model, SPENT, None, 1, processors=_ban())
    crit = _stream_criteria(SPENT, 2, enable_thinking=False, message="Go now")
    out = _stream(model, SPENT, crit, 1, processors=_ban())
    assert out[0][0] == plain[0][0]
    assert out[0][0] not in MESSAGE + [NL, END]
