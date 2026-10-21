"""A tokenizer with the real Beam token ids, for the harmony thinking-budget
tests. No model is needed.

``fixtures/beam_harmony_tokens.json`` holds the pieces the tests use, with
the ids the real GGUF tokenizer gives them, and the real encodings of the
test strings. ``encode`` takes the longest known piece at each position,
which reproduces those encodings (``test_the_beam_fixture_tokenizer_matches``
checks it).
"""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"
TABLE = json.loads((_FIXTURES / "beam_harmony_tokens.json").read_text())
PIECES: dict[str, int] = TABLE["pieces"]
VOCAB = 151936

START = "<|start|>assistant<|channel|>analysis<|message|>"
END = "<|end|>"


class BeamTokenizer:
    def __init__(self):
        self.chat_template = (_FIXTURES / "beam_template.jinja").read_text()
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
