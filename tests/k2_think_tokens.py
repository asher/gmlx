"""A K2 Horizon tokenizer stand-in for the think-pair tests. No model is
needed.

The control pieces carry the ids the real GGUF vocab gives them. Other text
falls back to one id per character. The chat template is the real one, so a
rendered generation prompt opens the pair its reasoning effort picks.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE = (Path(__file__).parent / "fixtures"
            / "k2_horizon_template.jinja").read_text()

PAIRS = {
    "high": ("<ifm|think>", "</ifm|think>"),
    "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
    "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
}

PIECES: dict[str, int] = {
    "<|ifm|begin_of_text|>": 0,
    "<|ifm|endoftext|>": 1,
    "<|ifm|im_start|>": 250018,
    "<|ifm|im_end|>": 250019,
    "<ifm|think>": 250029,
    "</ifm|think>": 250030,
    "<ifm|think_fast>": 250050,
    "</ifm|think_fast>": 250051,
    "<ifm|think_faster>": 250052,
    "</ifm|think_faster>": 250053,
}

_CHAR_BASE = 1000


class K2Tokenizer:
    chat_template = TEMPLATE
    eos_token_ids = [1, 250019]
    eos_token_id = 1

    def __init__(self):
        self._by_id = {v: k for k, v in PIECES.items()}
        self._longest = max(len(k) for k in PIECES)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for n in range(min(self._longest, len(text) - i), 0, -1):
                if text[i:i + n] in PIECES:
                    out.append(PIECES[text[i:i + n]])
                    i += n
                    break
            else:
                out.append(_CHAR_BASE + ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        return "".join(self._by_id.get(int(t)) or chr(int(t) - _CHAR_BASE)
                       for t in ids)


def generation_prompt(effort: str) -> str:
    """The real template's render of one user turn at ``effort``."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template(TEMPLATE).render(
        messages=[{"role": "user", "content": "hi"}],
        add_generation_prompt=True, bos_token="<|ifm|begin_of_text|>",
        reasoning_effort=effort)
