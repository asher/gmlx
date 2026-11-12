"""Think markers on serve, for every chat template the tests carry and for
every family whose base group sets markers.

A request that sets no markers takes them from the chat template, else
from the processor (the loader's mlx-lm wrapper). The same pair drives the
thinking budget, the ``response_format`` / XTC gate and the reasoning
split. Each template has a row in ``ROWS``: the pair it resolves, how its
switch turns thinking on and off, how the grammar meets the answer in each
mode, and the strings and merges its real vocab makes single tokens or
merges around a marker. tests/serve/test_think_marker_vocab_gguf.py checks
those last two against the real vocabs.

The gate kinds: "bare" (the grammar from the first token), "maybe" (the
first step also allows the first start token), "wait" (no mask until the
end marker) and "header" (no mask until the answer header).

Checks per template and server shape:

- (a) thinking off through the template switch: the first step;
- (a2) thinking off at the top level of the request: the first step;
- (b) thinking on: a scripted reply; only the answer is masked;
- (c) a budget of 4 with no markers configured: the plain criteria and
  the MTP hook;
- (d) the reply split, the stream splitter and the retire mirror.

A raw completion of each rendered prompt gates as its chat request.

A template file with no row fails ``test_every_template_has_a_row``, so a
new fixture enters every check.
"""

from __future__ import annotations

import copy
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("llguidance")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import think_vocab as tv  # noqa: E402

import gmlx  # noqa: E402
from gmlx.cache import retire_key  # noqa: E402
from gmlx.gen import profiles  # noqa: E402
from gmlx.gen import thinking_budget as tb  # noqa: E402
from gmlx.load import tokenizer as gguf_tokenizer  # noqa: E402
from gmlx.serve.patches import chat_behavior as sp_chat  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches.apc import install_retire_render_capture  # noqa: E402
from gmlx.upstream.patching import root  # noqa: E402

_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_ANTHROPIC = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
_RS = importlib.import_module("gmlx._vendor.mlx_vlm.server.responses_state")
_SCHEMAS = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
_PKG = importlib.import_module("gmlx._vendor.mlx_vlm.server")
_STRUCTURED = importlib.import_module("gmlx._vendor.mlx_vlm.structured")

FIXTURES = Path(__file__).parents[1] / "fixtures"
_VENDOR = Path(gmlx.__file__).parent / "_vendor"

MESSAGES = [{"role": "user", "content": "What is 3 + 4?"}]
REASON = "Add the two numbers first."
ANSWER = '{"answer": 7}'
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}},
           "required": ["answer"], "additionalProperties": False}
_FORMAT = {"type": "json_schema",
           "json_schema": {"name": "a", "schema": _SCHEMA}}

BARE, MAYBE, WAIT, HEADER = "bare", "maybe", "wait", "header"

_HARMONY = ("<|channel|>analysis<|message|>", "<|end|>")
_ATEM = ("<|start|>assistant to=self<|message|>", "<|eom|>")
_THINK = ("<think>", "</think>")
_GEMMA = ("<|channel>thought", "<channel|>")
_XTML = ("<|open|>think<|sep|>", "<|close|>think<|sep|>")
_MM = ("<mm:think>", "</mm:think>")

_HARMONY_SINGLES = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>",
                    "<|return|>", "<|constrain|>", "<|call|>")
_ATEM_SINGLES = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>")
_IM = ("<|im_start|>", "<|im_end|>")
_THINK_SINGLES = _THINK + _IM
_GEMMA_SINGLES = ("<|channel>", "<channel|>", "<|think|>", "<|turn>",
                  "<turn|>")
_XTML_SINGLES = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
_DS_SINGLES = _THINK + ("<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>")
_K2_SINGLES = ("<ifm|think>", "</ifm|think>", "<ifm|think_fast>",
               "</ifm|think_fast>", "<ifm|think_faster>",
               "</ifm|think_faster>", "<|ifm|im_start|>", "<|ifm|im_end|>")
# The Gemma vocab encodes "<think></think>" with the piece "></", and
# "first.</think>" with ".</".
_SLASH_MERGES = ("> <", ">< /", ". <", ".< /")

_ON = {"enable_thinking": True}
_OFF = {"enable_thinking": False}
# Kimi K3's answer header, after <|close|>think<|sep|> or in the prompt.
_XTML_ANSWER = "<|open|>response<|sep|>"


@dataclass(frozen=True)
class Row:
    """One template. ``pair`` is what a request with no markers resolves
    with thinking on (None: nothing). ``on`` / ``off`` are the template
    kwargs of each mode with its gate kind; ``off`` is None for a template
    with no off render. ``top_off`` is the gate kind of a request with a
    top-level ``enable_thinking: false``. ``vlm`` names the model type of
    the VLM processor shape."""

    model_type: str = "llama"
    pair: tuple | None = None
    on: tuple = ({}, BARE)
    off: tuple | None = None
    top_off: str = BARE
    singles: tuple = ()
    merges: tuple = ()
    vlm: str | None = None
    exclude: str | None = None


_IMAGE = "an image model's text-encoder prompt; serve never renders it"
_PYTHON = ("a template in Python that no GGUF load reaches: the loader "
           "builds its processor from the GGUF")

ROWS: dict[str, Row] = {
    "fixtures/beam_template.jinja": Row(
        "beam", _HARMONY, ({}, HEADER), None, HEADER, _HARMONY_SINGLES),
    "fixtures/deepseek_v4_template.jinja": Row(
        "deepseek_v4", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _DS_SINGLES),
    "fixtures/flint_template.jinja": Row(
        "flint", _THINK, ({}, MAYBE), None, BARE, _THINK_SINGLES),
    "fixtures/gemma4_12b_template.jinja": Row(
        "gemma4_text", _GEMMA, (_ON, MAYBE), (_OFF, BARE), BARE,
        _GEMMA_SINGLES, vlm="gemma4"),
    "fixtures/gemma4_template.jinja": Row(
        "gemma4_text", _GEMMA, (_ON, MAYBE), (_OFF, MAYBE), BARE,
        _GEMMA_SINGLES, vlm="gemma4"),
    "fixtures/glm5_template.jinja": Row(
        "glm_moe_dsa", _THINK, ({}, WAIT), None, WAIT, _THINK_SINGLES),
    "fixtures/gptoss_template.jinja": Row(
        "gpt_oss", _HARMONY, ({}, HEADER), None, HEADER, _HARMONY_SINGLES),
    "fixtures/granite_switch_systemone.jinja": Row(
        exclude="a systemone prompt, not a chat template"),
    "fixtures/granite_switch_template.jinja": Row(
        "granite_switch", None, (_ON, BARE), (_OFF, BARE), BARE,
        ("<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>")),
    "fixtures/image/flux2_klein_chat_template.jinja": Row(exclude=_IMAGE),
    "fixtures/image/flux3_klein_chat_template.jinja": Row(exclude=_IMAGE),
    "fixtures/image/moss_image_chat_template.jinja": Row(exclude=_IMAGE),
    "fixtures/image/z_image_turbo_chat_template.jinja": Row(exclude=_IMAGE),
    "fixtures/k2_horizon_template.jinja": Row(
        "k2_horizon", ("<ifm|think>", "</ifm|think>"), ({}, WAIT),
        ({"reasoning_effort": "low"}, WAIT), WAIT, _K2_SINGLES),
    "fixtures/kimi_k25_template.jinja": Row(
        "deepseek_v3", _THINK, ({"thinking": True}, WAIT),
        ({"thinking": False}, BARE), WAIT, _THINK_SINGLES, vlm="kimi_k25"),
    "fixtures/kimi_k3_template.jinja": Row(
        "kimi_k3", _XTML, ({"thinking": True}, HEADER),
        ({"thinking": False}, BARE), HEADER, _XTML_SINGLES),
    "fixtures/minicpmv46_template.jinja": Row(
        "qwen3", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="minicpmv4_6"),
    "fixtures/minicpmv47_template.jinja": Row(
        "qwen3", _THINK, (_ON, BARE), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="minicpmv4_7"),
    "fixtures/minimax_m31_template.jinja": Row(
        "minimax_m3", _MM, ({}, MAYBE), None, BARE, _MM),
    "fixtures/minimax_m31_thinking.jinja": Row(
        exclude="a template excerpt with no generation prompt"),
    "fixtures/minimax_m31_turns.jinja": Row(
        "minimax_m3", _MM, ({}, MAYBE), None, BARE, _MM),
    "fixtures/mistral3_template.jinja": Row(
        "ministral3", None, (_ON, BARE), (_OFF, BARE), BARE,
        ("[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")),
    "fixtures/mistral7b_v03_template.jinja": Row(
        "llama", None, ({}, BARE), None, BARE, ("[INST]", "[/INST]")),
    "fixtures/muse_spark_template.jinja": Row(
        "muse_spark", _ATEM, ({}, HEADER), None, HEADER, _ATEM_SINGLES,
        vlm="muse_spark"),
    "fixtures/muse_template.jinja": Row(
        "muse_glimmer", _ATEM, ({}, HEADER), None, HEADER, _ATEM_SINGLES),
    "fixtures/nemotron_bolt_template.jinja": Row(
        "gemma4_text", _THINK, (_ON, WAIT), (_OFF, BARE), BARE,
        _GEMMA_SINGLES + _IM, _SLASH_MERGES),
    "fixtures/nemotron_transcribe/chat_template.jinja": Row(
        "nemotron_h", _THINK, (_ON, WAIT), (_OFF, BARE), BARE,
        _THINK_SINGLES),
    "fixtures/qwen35_9b_template.jinja": Row(
        "qwen3_5", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="qwen3_5"),
    "fixtures/qwen35_template.jinja": Row(
        "qwen3_5", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="qwen3_5"),
    "fixtures/qwen36_template.jinja": Row(
        "qwen3_5", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="qwen3_5"),
    "fixtures/qwen38_template.jinja": Row(
        "qwen3_5", _THINK, (_ON, WAIT), (_OFF, BARE), BARE, _THINK_SINGLES,
        vlm="qwen3_5"),
    "fixtures/qwen3_template.jinja": Row(
        "qwen3", _THINK, (_ON, MAYBE), (_OFF, BARE), BARE, _THINK_SINGLES),
    "fixtures/ricursive_template.jinja": Row(
        "ricursive", _THINK, ({}, MAYBE), None, BARE, _THINK_SINGLES),
    "fixtures/smollm2_template.jinja": Row(
        "llama", None, ({}, BARE), None, BARE, _IM),
    "bundled/deepseek_v41": Row(
        "deepseek_v41", _THINK, (_ON, WAIT), (_OFF, BARE), BARE,
        _DS_SINGLES),
    "python/deepseek_v4": Row(exclude=_PYTHON),
    "python/hrm_text": Row(exclude=_PYTHON),
    "python/mlx_lm/deepseek_v32": Row(exclude=_PYTHON),
}


def _template_sources() -> dict[str, str | None]:
    """Every template the tests or gmlx carry, by key. The Python ones are
    found by scanning, so a new one needs a row."""
    out: dict[str, str | None] = {}
    for p in sorted(FIXTURES.rglob("*.jinja")):
        out["fixtures/" + p.relative_to(FIXTURES).as_posix()] = p.read_text()
    for model_type in gguf_tokenizer.BUNDLED_CHAT_TEMPLATES:
        out[f"bundled/{model_type}"] = gguf_tokenizer.bundled_chat_template(
            model_type)
    for p in sorted((_VENDOR / "mlx_vlm/models").glob("*/processing_*.py")):
        if re.search(r"^DEFAULT_CHAT_TEMPLATE\s*=", p.read_text(), re.M):
            out[f"python/{p.parent.name}"] = None
    for p in sorted((_VENDOR / "mlx_lm/chat_templates").glob("*.py")):
        if p.stem != "__init__":
            out[f"python/mlx_lm/{p.stem}"] = None
    return out


TEMPLATES = _template_sources()
_SERVED = [k for k in sorted(TEMPLATES) if k in ROWS and not ROWS[k].exclude]
_CASES = [pytest.param(k, "text", id=f"{k}-text") for k in _SERVED] + [
    pytest.param(k, "vlm", id=f"{k}-vlm") for k in _SERVED if ROWS[k].vlm]


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_has_a_row(key):
    assert key in ROWS, f"add a row to ROWS for {key}"


@pytest.mark.parametrize("key", sorted(ROWS))
def test_every_row_has_a_template(key):
    assert key in TEMPLATES, f"{key} is gone: drop its row"


def test_every_marker_family_has_a_row():
    marked = {name for name, fam in profiles.FAMILIES.items()
              if fam["base"].get("sampling", {}).get("thinking_start_token")}
    assert marked == set(FAMILY_ROWS)


# -- the serve seams -----------------------------------------------------------


def _cpu_mask(logits, mask):
    from gmlx.spec.grammar import apply_masks

    if logits.ndim == 1:
        logits = logits[None, :]
    return apply_masks(logits, np.array(mask))


@pytest.fixture
def serve(monkeypatch):
    """The gen-args transforms, the render seam, the retire capture and the
    splitters as serve installs them. The grammar masks on the CPU."""
    seams = ([(t, "apply_chat_template")
              for t in sp_common._render_target_modules()]
             + [(_RS.ThinkingStreamState, "__init__"),
                (_RS.ThinkingStreamState, "feed"),
                (_APP, "_split_thinking_text"), (_RS, "_split_thinking"),
                (_GEN.ResponseGenerator, "_preprocess_request"),
                (_GEN.ResponseGenerator, "_make_thinking_budget_criteria"),
                (_PKG, "_infer_tool_parser_from_processor"),
                (_GEN.GenerationArguments, "to_template_kwargs"),
                (_APP, "_build_gen_args"), (_OPENAI, "_build_gen_args"),
                (_ANTHROPIC, "_build_gen_args"),
                (getattr(_APP, "_protocol_deps", None), "build_gen_args")])
    for obj, attr in seams:
        if hasattr(obj, attr):
            monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setattr(_STRUCTURED, "_apply_llguidance_mask", _cpu_mask)
    monkeypatch.delenv("GMLX_APC_RETIRE_LCP", raising=False)
    tokens = (sp_chat._LAST_GEN_ARGS.set(None),
              sp_chat._LAST_RENDERED_PROMPT.set(None))
    retire_key._reset_for_tests()
    sp_chat.install_chat_template_kwargs()
    install_retire_render_capture()
    sp_chat.install_stream_thinking_seed()
    sp_chat.install_thinking_budget_fix()
    yield
    retire_key._reset_for_tests()
    sp_chat._LAST_GEN_ARGS.reset(tokens[0])
    sp_chat._LAST_RENDERED_PROMPT.reset(tokens[1])


@dataclass
class Served:
    rg: object
    tok: object
    model_type: str


_MM_META = {"clip.vision.patch_size": 16, "clip.vision.image_size": 224,
            "clip.vision.projection_dim": 8,
            "clip.vision.image_mean": [0.5] * 3,
            "clip.vision.image_std": [0.5] * 3,
            "clip.vision.spatial_merge_size": 2}


def _served(template, row, shape) -> Served:
    raw = tv.tokenizer(template, singles=row.singles, merges=row.merges,
                       model_type=row.model_type)
    if shape == "text":
        proc, model_type = tv.text_processor(raw), row.model_type
    else:
        from gmlx.load.vlm import _synthesize_vlm_processor

        proc = _synthesize_vlm_processor(row.vlm, raw, _MM_META,
                                         {"model_type": row.vlm})
        model_type = row.vlm
    rg = _GEN.ResponseGenerator.__new__(_GEN.ResponseGenerator)
    rg.tokenizer, rg.processor = proc.tokenizer, proc
    return Served(rg, proc.tokenizer, model_type)


@dataclass
class Request:
    args: object
    prompt: str
    ids: list


def _request(s: Served, *, kwargs=None, body=None, markers=None) -> Request:
    """A chat request through ``_build_gen_args`` and the render seam."""
    extra = dict(body or {})
    if kwargs:
        extra["chat_template_kwargs"] = kwargs
    req = _SCHEMAS.ChatRequest(model="m", messages=MESSAGES, max_tokens=8,
                               response_format=_FORMAT, **extra)
    args = _APP._build_gen_args(req, s.rg.processor)
    if markers:
        args.thinking_start_token, args.thinking_end_token = markers
    prompt = _OPENAI.apply_chat_template(
        s.rg.processor, {"model_type": s.model_type}, MESSAGES,
        **args.to_template_kwargs())
    assert isinstance(prompt, str)
    return Request(args, prompt, tv.ids(s.tok, prompt))


def _gate(s: Served, r: Request):
    make = root(_GEN.ResponseGenerator._make_logits_processors)
    (proc,) = make(s.rg, r.args, mx.array([r.ids]))
    return proc


def _allowed_rows(proc, prompt_ids, gen, vocab):
    """The batch loop: the prompt, then each sampled token. Per generated
    step, which tokens the processed logits allow."""
    logits = mx.zeros((1, vocab))
    rows = [np.isfinite(np.array(proc(mx.array(prompt_ids), logits)))[0]]
    for t in gen[:-1]:
        rows.append(np.isfinite(np.array(
            proc.process_last_token(t, logits)))[0])
    return rows


def _vocab(proc):
    inner = getattr(proc, "processor", proc)
    return inner.llg_tokenizer.vocab_size


def _first_step(s, r, kind, start):
    proc = _gate(s, r)
    (row,) = _allowed_rows(proc, r.ids, [0], _vocab(proc))
    x, brace = tv.ids(s.tok, "x")[0], tv.ids(s.tok, "{")[0]
    if kind == BARE and s.model_type == "qwen3_5":
        # The fused greedy path needs the bare grammar.
        assert type(proc) is _STRUCTURED.LLGuidanceLogitsProcessor
    if kind in (WAIT, HEADER):
        assert row.all()
        return
    first = tv.ids(s.tok, start)[0] if start else None
    answer = [int(i) for i in np.nonzero(row)[0] if i != first]
    # A real vocab may allow only a merged piece such as '{"'.
    assert answer and not row[x]
    assert all(s.tok.decode([i]).startswith("{") for i in answer)
    if kind == MAYBE:
        assert row[first]
    elif first is not None and first != brace:
        assert not row[first]


def _reply_parts(row, pair, kind):
    """``(prefix, answer)``: the scripted reply before its answer."""
    if kind == HEADER:
        if pair == _XTML:
            return REASON + pair[1] + _XTML_ANSWER, ANSWER
        if pair == _HARMONY:
            return (pair[0] + REASON + pair[1]
                    + "<|start|>assistant<|channel|>final<|message|>", ANSWER)
        return (" to=self<|message|>" + REASON + pair[1]
                + "<|start|>assistant to=user<|message|>", ANSWER)
    if kind == WAIT:
        return REASON + "\n" + pair[1], ANSWER
    if kind == MAYBE:
        return pair[0] + REASON + "\n" + pair[1], ANSWER
    return "", ANSWER


def _scripted(s, r, kind, pair, prefix, answer):
    proc = _gate(s, r)
    pre = tv.ids(s.tok, prefix)
    gen = tv.ids(s.tok, prefix + answer)
    assert gen[:len(pre)] == pre
    rows = _allowed_rows(proc, r.ids, gen, _vocab(proc))
    forced = len(tv.ids(s.tok, pair[0])) if kind == MAYBE else 0
    for k, t in enumerate(gen):
        assert rows[k][t], f"step {k} refuses the scripted token"
        if k >= len(pre) or kind == BARE:
            assert not rows[k].all(), f"step {k} is not masked"
        elif kind == MAYBE and k == 0:
            assert not rows[k].all()
        elif kind == MAYBE and k < forced:
            assert rows[k].sum() == 1, f"step {k} does not force the start"
        else:
            assert rows[k].all(), f"step {k} is masked inside reasoning"


# -- (a) to (d) per template --------------------------------------------------


@pytest.mark.parametrize("key,shape", _CASES)
def test_a_thinking_off_through_the_template(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    kwargs, kind = row.off or row.on
    _first_step(s, _request(s, kwargs=kwargs), kind,
                row.pair[0] if row.pair else None)


@pytest.mark.parametrize("key,shape", _CASES)
def test_a2_thinking_off_at_the_top_level(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    r = _request(s, body={"enable_thinking": False})
    _first_step(s, r, row.top_off, row.pair[0] if row.pair else None)


@pytest.mark.parametrize("key,shape", _CASES)
def test_b_thinking_on_masks_only_the_answer(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    kwargs, kind = row.on
    r = _request(s, kwargs=kwargs)
    prefix, answer = _reply_parts(row, row.pair, kind)
    _scripted(s, r, kind, row.pair, prefix, answer)
    if kind == MAYBE:
        _scripted(s, _request(s, kwargs=kwargs), kind, row.pair, "", ANSWER)


def _seq_in(run, seq) -> bool:
    return tb.rfind_seq(list(run), list(seq)) >= 0


def _budget_runs(s, r):
    from gmlx.serve.patches.mtp_thinking import _build_hook

    r.args.thinking_budget = 4
    crit = _GEN.ResponseGenerator._make_thinking_budget_criteria(
        s.rg, r.args, mx.array([r.ids]))
    hook = _build_hook(s.rg, r.args, mx.array([r.ids]))
    return crit, hook


def _opens_at(s, prefix):
    """The number of reply tokens before the reasoning text."""
    return len(tv.ids(s.tok, prefix[:prefix.index(REASON)]))


@pytest.mark.parametrize("key,shape", _CASES)
def test_c_a_budget_with_no_markers_resolves_the_pair(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    kwargs, kind = row.on
    r = _request(s, kwargs=kwargs)
    crit, hook = _budget_runs(s, r)
    if row.pair is None:
        assert crit._end_seq == tuple(tv.ids(s.tok, _THINK[1]))
        assert hook is None
        return
    start, end = (tuple(tv.ids(s.tok, m)) for m in row.pair)
    assert (crit._start_seq, crit._end_seq) == (start, end)
    assert (hook.start_seq, hook.end_seq) == (start, end)
    # The prompt opens the block when the reply starts with reasoning.
    prefix = _reply_parts(row, row.pair, kind)[0]
    assert crit.in_thinking is hook.in_thinking is prefix.startswith(REASON)
    if kind == BARE:
        prefix = _reply_parts(row, row.pair, MAYBE)[0]
    gen = tv.ids(s.tok, prefix)
    replies = _closes_at_the_budget(crit, hook, gen, _opens_at(s, prefix), end)
    if kind == HEADER and row.pair != _ATEM:
        for reply in replies:
            _answer_follows_the_close(s, _request(s, kwargs=kwargs), reply)


def _answer_follows_the_close(s, r, reply):
    """The forced close writes the answer header, so the grammar masks the
    next token."""
    proc = _gate(s, r)
    gen = list(reply) + tv.ids(s.tok, ANSWER)
    rows = _allowed_rows(proc, r.ids, gen, _vocab(proc))
    assert all(rows[k].all() for k in range(len(reply)))
    assert not rows[len(reply)].all() and rows[len(reply)][gen[len(reply)]]


def _closes_at_the_budget(crit, hook, gen, opens, end):
    """The plain criteria and the MTP hook close the block on the same
    reply token, after 4 reasoning tokens, with the end marker. Returns
    the reply each one writes up to the end of its forced run."""
    assert crit.thinking_token_count == hook.count
    want = opens + 4 - crit.thinking_token_count - 1
    for i, t in enumerate(gen):
        hook.observe(t)
        taken, forced = hook.take_forced(), crit(t)
        if taken is not None or forced is not None:
            assert (i, forced is not None) == (want, True)
            assert _seq_in(taken, end)
            break
    else:
        pytest.fail("the budget never closed the block")
    run = [crit.forced_token_id]
    while (forced := crit(run[-1])) is not None:
        run.append(forced)
    assert _seq_in(run, end)
    return list(gen[:i + 1]) + run, list(gen[:i + 1]) + list(taken)


def _splits(s, r, text):
    split = _APP._split_thinking_text(
        text, r.args.thinking_start_token, r.args.thinking_end_token, False)
    state = _RS.ThinkingStreamState(r.args.enable_thinking,
                                    r.args.thinking_start_token,
                                    r.args.thinking_end_token)
    reasoning, content = [], []
    for i in range(0, len(text), 3):
        d = state.feed(text[i:i + 3])
        reasoning.append(d.reasoning or "")
        content.append(d.content or "")
    d = state.feed("", last=True)
    reasoning.append(d.reasoning or "")
    content.append(d.content or "")
    return split, ("".join(reasoning), "".join(content))


def _on_reply(row):
    """The scripted reply with thinking on and the reasoning it holds."""
    kind = row.on[1]
    if row.pair is None or kind == BARE:
        return ANSWER, ""
    prefix, answer = _reply_parts(row, row.pair, kind)
    return prefix + answer, REASON


@pytest.mark.parametrize("key,shape", _CASES)
def test_d_the_reply_split_takes_the_resolved_pair(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    text, want = _on_reply(row)
    split, _stream = _splits(s, _request(s, kwargs=row.on[0]), text)
    assert ((split[0] or "").strip(), split[1].strip()) == (want, ANSWER)
    if row.off is not None and row.off[1] != WAIT:
        split, _stream = _splits(s, _request(s, kwargs=row.off[0]), ANSWER)
        assert (split[0], split[1].strip()) == (None, ANSWER)


def _retired(r, text):
    """``(reasoning, content)`` of the retire mirror's assistant message.
    It runs at retirement, outside the request's context."""
    ctx = retire_key._TEXT_MEMO[r.prompt]
    tokens = (sp_chat._LAST_GEN_ARGS.set(None),
              sp_chat._LAST_RENDERED_PROMPT.set(None))
    try:
        msg = retire_key.build_assistant_message(ctx, text)
    finally:
        sp_chat._LAST_GEN_ARGS.reset(tokens[0])
        sp_chat._LAST_RENDERED_PROMPT.reset(tokens[1])
    return (msg.get("reasoning_content") or "").strip(), msg["content"].strip()


@pytest.mark.parametrize("key,shape", _CASES)
def test_d_the_retire_mirror_splits_as_the_reply_split(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    text, want = _on_reply(row)
    r = _request(s, kwargs=row.on[0])
    split, _stream = _splits(s, r, text)
    assert ((split[0] or "").strip(), split[1].strip()) == (want, ANSWER)
    assert _retired(r, text) == (want, ANSWER)
    if row.off is not None and row.off[1] != WAIT:
        r = _request(s, kwargs=row.off[0])
        assert _retired(r, ANSWER) == ("", ANSWER)


@pytest.mark.parametrize("key,shape", _CASES)
def test_d_the_stream_split_takes_the_resolved_pair(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    text, want = _on_reply(row)
    _split, stream = _splits(s, _request(s, kwargs=row.on[0]), text)
    assert (stream[0].strip(), stream[1].strip()) == (want, ANSWER)
    if row.off is not None and row.off[1] != WAIT:
        _split, stream = _splits(s, _request(s, kwargs=row.off[0]), ANSWER)
        assert (stream[0].strip(), stream[1].strip()) == ("", ANSWER)


# -- a raw completion, a seeded first step, one configured marker -------------


def _raw(r: Request) -> Request:
    """``r`` as a raw completion of its prompt: no render kept markers."""
    args = copy.copy(r.args)
    del args._kq_think
    return Request(args, r.prompt, r.ids)


@pytest.mark.parametrize("key,shape", _CASES)
def test_a_raw_completion_gates_as_its_chat_request(serve, key, shape,
                                                    monkeypatch):
    """With no render, the engine resolves the markers from the prompt ids:
    one decode, in the budget hook under the tokenizer lock, and none on
    the GPU worker. The gate then matches the chat request's."""
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    decodes = []
    prompt_text = tb._prompt_text
    monkeypatch.setattr(tb, "_prompt_text", lambda p, t: (
        decodes.append(p), prompt_text(p, t))[1])
    for kwargs, _kind in [row.on] + ([row.off] if row.off else []):
        chat = _request(s, kwargs=kwargs)
        raw = _raw(_request(s, kwargs=kwargs))
        proc = _gate(s, chat)
        want = _allowed_rows(proc, chat.ids, [0], _vocab(proc))
        decodes.clear()
        _budget_runs(s, raw)
        assert len(decodes) == 1
        proc = _gate(s, raw)
        assert len(decodes) == 1
        got = _allowed_rows(proc, raw.ids, [0], _vocab(proc))
        assert (got[0] == want[0]).all()


def _seeded_rows(proc, prompt_ids, gen, vocab):
    """As ``_allowed_rows``, when the batch loop hands the prompt's last
    token to ``process_last_token`` before the first step."""
    logits = mx.zeros((1, vocab))
    return [np.isfinite(np.array(proc.process_last_token(t, logits)))[0]
            for t in [prompt_ids[-1]] + gen[:-1]]


_MAYBE_CASES = [c for c in _CASES if ROWS[c.values[0]].on[1] == MAYBE]


@pytest.mark.parametrize("key,shape", _MAYBE_CASES)
def test_b_a_seeded_maybe_row_masks_as_after_the_prompt(serve, key, shape):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, shape)
    kwargs = row.on[0]
    for prefix in (_reply_parts(row, row.pair, MAYBE)[0], ""):
        gen = tv.ids(s.tok, prefix + ANSWER)
        r = _request(s, kwargs=kwargs)
        proc = _gate(s, r)
        want = _allowed_rows(proc, r.ids, gen, _vocab(proc))
        r = _request(s, kwargs=kwargs)
        proc = _gate(s, r)
        got = _seeded_rows(proc, r.ids, gen, _vocab(proc))
        for k, (a, b) in enumerate(zip(got, want)):
            assert (a == b).all(), f"step {k} differs"


@pytest.mark.parametrize("key,kwargs,body,kind", [
    # The resolved pair finds no thinking: bare. One configured marker
    # keeps the old gate, which waits for </think>.
    ("fixtures/smollm2_template.jinja", {}, None, WAIT),
    ("fixtures/smollm2_template.jinja", {}, {"enable_thinking": False}, BARE),
    # The off prompt ends with </think>: bare, as before.
    ("fixtures/qwen3_template.jinja", _OFF, None, BARE),
    # The prompt opens <think> and the old gate waits, as the resolved
    # one does.
    ("fixtures/qwen35_9b_template.jinja", _ON, None, WAIT),
])
def test_a_one_configured_marker_keeps_the_old_gate(serve, key, kwargs, body,
                                                    kind):
    row = ROWS[key]
    s = _served(TEMPLATES[key], row, "text")
    r = _request(s, kwargs=kwargs, body=body, markers=(None, _THINK[1]))
    assert r.args._kq_think.partial
    _first_step(s, r, kind, _THINK[0])


def _native_k3() -> Served:
    """An MLX-format Kimi K3: its processor renders in Python and carries
    no template text."""
    from gmlx._vendor.mlx_vlm.models.kimi_k3.processing_kimi_k3 import (
        KimiK3Processor)

    proc = KimiK3Processor(tokenizer=tv.tokenizer(None, singles=_XTML_SINGLES))
    rg = _GEN.ResponseGenerator.__new__(_GEN.ResponseGenerator)
    rg.tokenizer, rg.processor = proc.tokenizer, proc
    return Served(rg, proc.tokenizer, "kimi_k3")


@pytest.mark.parametrize("thinking,kind", [(True, HEADER), (False, BARE)])
def test_the_native_k3_renderer_resolves_the_xtml_pair(serve, thinking, kind):
    s = _native_k3()
    r = _request(s, kwargs={"thinking": thinking})
    assert r.prompt.endswith(_XTML[0] if thinking else _XTML_ANSWER)
    assert r.args._kq_think.pair == _XTML
    _first_step(s, r, kind, _XTML[0])
    if thinking:
        _scripted(s, _request(s, kwargs={"thinking": True}), kind, _XTML,
                  REASON + _XTML[1] + _XTML_ANSWER, ANSWER)
        text = (REASON + _XTML[1] + _XTML_ANSWER + ANSWER
                + "<|close|>response<|sep|>")
        _split, stream = _splits(s, r, text)
        assert stream == (REASON, ANSWER)
        crit, hook = _budget_runs(s, _request(s, kwargs={"thinking": True}))
        end = tuple(tv.ids(s.tok, _XTML[1]))
        assert crit._end_seq == hook.end_seq == end


# -- families with markers in their base group --------------------------------


@dataclass(frozen=True)
class FamilyRow:
    singles: tuple
    header: bool = False


FAMILY_ROWS = {
    "gpt-oss": FamilyRow(_HARMONY_SINGLES, header=True),
    "beam": FamilyRow(_HARMONY_SINGLES, header=True),
    "muse": FamilyRow(_ATEM_SINGLES, header=True),
    "muse-spark": FamilyRow(_ATEM_SINGLES, header=True),
    "kimi": FamilyRow(_XTML_SINGLES, header=True),
    "minimax-m3": FamilyRow(_MM),
    "hy3": FamilyRow(("<think:opensource>", "</think:opensource>")),
    "hy4": FamilyRow(("<think:6124c78e>", "</think:6124c78e>")),
}
_ECHO = "{{ messages[-1]['content'] }}"


def _family_pair(family):
    sampling = profiles.FAMILIES[family]["base"]["sampling"]
    return sampling["thinking_start_token"], sampling["thinking_end_token"]


def _family_prompts(family):
    """Prompt -> (gate kind, top-level-off gate kind, scripted prefix)."""
    start, end = _family_pair(family)
    if start == _XTML[0]:
        # The header gate opens on a prompt that ends with the header.
        return {
            "open": ("Q" + start + "\n", HEADER, HEADER,
                     REASON + "\n" + end + _XTML_ANSWER),
            "closed": ("Q" + start + "\n\n" + end, HEADER, HEADER,
                       _XTML_ANSWER),
            "plain": ("Q\n", HEADER, HEADER,
                      start + REASON + "\n" + end + _XTML_ANSWER),
            "answer": ("Q" + _XTML_ANSWER, BARE, BARE, ""),
        }
    if FAMILY_ROWS[family].header:
        if "channel" in start:
            prompt = "Q<|end|><|start|>assistant"
            prefix = ("<|channel|>analysis<|message|>" + REASON + end
                      + "<|start|>assistant<|channel|>final<|message|>")
        else:
            prompt = "Q<|eot|><|start|>assistant"
            prefix = (" to=self<|message|>" + REASON + end
                      + "<|start|>assistant to=user<|message|>")
        return {"header": (prompt, HEADER, HEADER, prefix)}
    return {
        "open": ("Q" + start + "\n", WAIT, WAIT, REASON + "\n" + end),
        "closed": ("Q" + start + "\n\n" + end + "\n\n", BARE, BARE, ""),
        "plain": ("Q\n", MAYBE, BARE, start + REASON + "\n" + end),
    }


_FAMILY_CASES = [pytest.param(f, p, id=f"{f}-{p}")
                 for f in FAMILY_ROWS for p in _family_prompts(f)]


def _family_request(family, prompt, body=None):
    row = Row("llama", _family_pair(family), singles=FAMILY_ROWS[family].singles)
    s = _served(_ECHO, row, "text")
    messages = [{"role": "user", "content": prompt}]
    req = _SCHEMAS.ChatRequest(model="m", messages=messages, max_tokens=8,
                               response_format=_FORMAT, **(body or {}))
    args = _APP._build_gen_args(req, s.rg.processor)
    args.thinking_start_token, args.thinking_end_token = _family_pair(family)
    text = _OPENAI.apply_chat_template(
        s.rg.processor, {"model_type": "llama"}, messages,
        **args.to_template_kwargs())
    assert text == prompt
    return s, Request(args, text, tv.ids(s.tok, text))


@pytest.mark.parametrize("family,name", _FAMILY_CASES)
def test_a_family_marker_gate(serve, family, name):
    prompt, kind, top_off, prefix = _family_prompts(family)[name]
    start = _family_pair(family)[0]
    s, r = _family_request(family, prompt)
    _first_step(s, r, kind, start)
    s, r = _family_request(family, prompt, {"enable_thinking": False})
    _first_step(s, r, top_off, start)
    s, r = _family_request(family, prompt)
    _scripted(s, r, kind, _family_pair(family), prefix if kind != BARE else "",
              ANSWER)


@pytest.mark.parametrize("family,name", _FAMILY_CASES)
def test_c_a_family_budget_takes_the_family_pair(serve, family, name):
    prompt, kind, _top, prefix = _family_prompts(family)[name]
    pair = _family_pair(family)
    s, r = _family_request(family, prompt)
    crit, hook = _budget_runs(s, r)
    start, end = (tuple(tv.ids(s.tok, m)) for m in pair)
    assert (crit._start_seq, crit._end_seq) == (start, end)
    assert (hook.start_seq, hook.end_seq) == (start, end)
    assert crit.in_thinking is hook.in_thinking is prefix.startswith(REASON)
    if REASON not in prefix:
        prefix = pair[0] + REASON + "\n" + pair[1]
    _closes_at_the_budget(crit, hook, tv.ids(s.tok, prefix),
                          _opens_at(s, prefix), end)


@pytest.mark.parametrize("family,name", _FAMILY_CASES)
def test_a_family_marker_split(serve, family, name):
    _prompt, kind, _top, prefix = _family_prompts(family)[name]
    s, r = _family_request(family, _prompt)
    text = prefix + ANSWER
    split, stream = _splits(s, r, text)
    want = REASON if REASON in prefix else ""
    assert ((split[0] or "").strip(), split[1].strip()) == (want, ANSWER)
    assert (stream[0].strip(), stream[1].strip()) == (want, ANSWER)
    assert _retired(r, text) == (want, ANSWER)
