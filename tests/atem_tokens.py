"""A tokenizer with the real Muse Spark and Muse Glimmer token ids, for the
ATEM thinking-budget tests. No model is needed.

``fixtures/muse_atem_tokens.json`` holds the pieces the tests use, with the
ids the real GGUF headers give them, and the real encodings of the test
strings. ``encode`` takes the longest known piece at each position, which
reproduces those encodings (``test_the_fixture_tokenizer_matches_the_real_one``
checks it). Other text falls back to single characters, which are real
single-byte tokens too.
"""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"
TABLE = json.loads((_FIXTURES / "muse_atem_tokens.json").read_text())
PIECES: dict[str, int] = TABLE["pieces"]

SELF_OPEN = "<|start|>assistant to=self<|message|>"
EOM = "<|eom|>"


class AtemTokenizer:
    """``template`` is the chat template to report: "spark" (the GGUF's) or
    "glimmer" (the fixture Glimmer template)."""

    def __init__(self, template: str = "spark"):
        name = {"spark": "muse_spark_template.jinja", "glimmer": "muse_template.jinja"}[
            template
        ]
        self.chat_template = (_FIXTURES / name).read_text()
        self.eos_token_ids = list(TABLE["eos_token_ids"])
        self.eos_token_id = self.eos_token_ids[0]
        self._by_id = {v: k for k, v in PIECES.items()}
        self._longest = max(len(k) for k in PIECES)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for n in range(min(self._longest, len(text) - i), 0, -1):
                tid = PIECES.get(text[i : i + n])
                if tid is not None:
                    out.append(tid)
                    i += n
                    break
            else:
                raise KeyError(f"no fixture piece for {text[i]!r}")
        return out

    def decode(self, token_ids, **kw):
        return "".join(self._by_id[int(t)] for t in token_ids)

    def convert_tokens_to_ids(self, token):
        return PIECES[token]
