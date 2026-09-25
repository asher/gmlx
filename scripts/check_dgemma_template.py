#!/usr/bin/env python
"""Check that a DiffusionGemma GGUF renders structured-decision prompts to
the same token ids as the upstream Hugging Face tokenizer and template.

    python scripts/check_dgemma_template.py DIFFUSIONGEMMA.gguf \
        [--hf google/diffusiongemma-26B-A4B-it]

The gmlx side builds the tokenizer from the GGUF header the way the loader
does and renders through the same helper the ``/v1/systemone`` route uses.
The reference side is ``AutoTokenizer`` on the HF repo or a local directory,
called the way the vLLM structured-diffusion example calls it. The check
covers plain and thinking prompts, a prompt continued by answer lines, the
thought tags and every answer template the decision logic resolves. Exit
status 0 means every case matched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from gmlx.systemone import TemplateResolver, jev_schema, jev_state, system_text
from gmlx.systemone.engine import ChatTokens
from gmlx.systemone.template import FORMATS, answer_text

_REQUEST = {
    "state": {
        "ticket": "Everything is down and we have a demo at noon. "
                  "Unicode check: caf\u00e9, \u65e5\u672c\u8a9e, \U0001f525.",
        "customer": {"plan": "enterprise", "seats": 1200},
    },
    "instructions": "You triage support tickets.",
    "questions": {
        "urgent": {"type": "noul",
                   "instructions": "Does the customer need a reply within the hour?",
                   "criteria": {"true": "a reply is needed within the hour",
                                "false": "a reply can wait"}},
        "area": {"type": "choice", "instructions": "Which team owns the ticket?",
                 "criteria": {"infra": "outages and latency",
                              "billing": "invoices and plans",
                              "product": "features and bugs"}},
        "severity": {"type": "score", "instructions": "How severe is it?",
                     "criteria": ["low", "medium", "high", "critical"]},
        "refund": {"type": "noul", "instructions": "Is a refund requested?",
                   "depends_on": ["area"], "ask_if": {"area": ["billing"]}},
    },
}


def gguf_tokens(path: str, state: str) -> ChatTokens:
    """The route's tokenizer pair, built from the GGUF header alone."""
    from gmlx.distill.tokens import _gguf_kv
    from gmlx.load.config_synth import GGUF_ARCH_TO_MODEL_TYPE
    from gmlx.load.tokenizer import (
        bundled_chat_template_for_arch,
        finish_gguf_tokenizer,
        load_tokenizer_from_gguf,
    )
    from gmlx.serve.bridge_vlm import _make_text_processor

    kv = _gguf_kv(path)
    arch = kv["general.architecture"]
    model_type = GGUF_ARCH_TO_MODEL_TYPE.get(arch)
    if model_type != "diffusion_gemma":
        sys.exit(f"{path}: architecture {arch!r} is not DiffusionGemma")
    raw = load_tokenizer_from_gguf(
        kv, arch, chat_template_override=bundled_chat_template_for_arch(arch))
    finish_gguf_tokenizer(raw, model_type)
    wrapper_cls = importlib.import_module("mlx_lm.tokenizer_utils").TokenizerWrapper
    wrapper = wrapper_cls(raw, eos_token_ids=getattr(raw, "_gguf_eos_token_ids", None))
    return ChatTokens(_make_text_processor(wrapper), state)


class HfTokens:
    """The vLLM example's ``enc`` and ``chat_prompt_ids`` on the HF tokenizer."""

    def __init__(self, ref: str, state: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(ref)
        self.state_text = state

    def enc(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def chat_ids(self, sys_text: str, thinking: bool) -> list[int]:
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": self.state_text}]
        out = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return [int(t) for t in ids]


def compare(name: str, ours, theirs) -> bool:
    ours, theirs = list(ours), list(theirs)
    if ours == theirs:
        print(f"same    {name} ({len(ours)} ids)")
        return True
    at = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b),
              min(len(ours), len(theirs)))
    print(f"DIFFER  {name}: gmlx {len(ours)} ids, hf {len(theirs)} ids, "
          f"first difference at {at}")
    print(f"        gmlx {ours[max(0, at - 4):at + 6]}")
    print(f"        hf   {theirs[max(0, at - 4):at + 6]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", help="the DiffusionGemma GGUF")
    ap.add_argument("--hf", default="google/diffusiongemma-26B-A4B-it",
                    help="HF repo id or local directory holding the reference tokenizer")
    ap.add_argument("--canvas", type=int, default=64, help="the served canvas length")
    a = ap.parse_args()

    schema = jev_schema(_REQUEST)
    state = jev_state(_REQUEST)
    ours = gguf_tokens(a.gguf, state)
    theirs = HfTokens(a.hf, state)
    ok = True

    full = system_text(schema)
    chunked = system_text(schema, chunked=True)
    for name, text in (("system", full), ("system, chunked", chunked)):
        ok &= compare(f"{name}, plain prompt", ours.chat_ids(text, False),
                      theirs.chat_ids(text, False))
        ok &= compare(f"{name}, thinking prompt", ours.chat_ids(text, True),
                      theirs.chat_ids(text, True))

    ok &= compare("state text", ours.enc(state), theirs.enc(state))

    r_ours = TemplateResolver(ours.enc, a.canvas)
    r_theirs = TemplateResolver(theirs.enc, a.canvas)
    for tag in ("thought_open", "thought_close", "scaffold"):
        ok &= compare(tag, getattr(r_ours, tag), getattr(r_theirs, tag))

    qs = schema["questions"]
    fmt = schema.get("format", "lines")
    join = FORMATS[fmt][0]
    for head_name, head in (("scaffold head", r_ours.scaffold), ("empty head", ())):
        t_ours, s_ours = r_ours.resolve(qs, list(head), "", fmt)
        t_theirs, s_theirs = r_theirs.resolve(qs, list(head), "", fmt)
        ok &= compare(f"answer template, {head_name}", t_ours, t_theirs)
        ok &= compare(f"slot positions, {head_name}",
                      [s.pos for s in s_ours], [s.pos for s in s_theirs])
        ok &= compare(f"slot label ids, {head_name}",
                      [i for s in s_ours for i in s.label_ids],
                      [i for s in s_theirs for i in s.label_ids])

    lines = answer_text(qs[:2], [0, 1], fmt)
    base = ours.chat_ids(full, False) + list(r_ours.scaffold)
    ok &= compare("continued prompt",
                  base + ours.enc(lines),
                  theirs.chat_ids(full, False) + list(r_theirs.scaffold) + theirs.enc(lines))
    t_ours, _ = r_ours.resolve(qs[2:], [], join, fmt)
    t_theirs, _ = r_theirs.resolve(qs[2:], [], join, fmt)
    ok &= compare("conditioned answer template", t_ours, t_theirs)

    print(json.dumps({"gguf": a.gguf, "reference": a.hf, "all_same": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
