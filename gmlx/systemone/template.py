# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM examples/features/structured_diffusion/structured_server.py
# at ab3de6edf2 and modified for gmlx (Apache-2.0; see
# licenses/vllm-LICENSE).
"""Answer templates: the system text, the answer lines, and the slot
resolution against a tokenizer.

``TemplateResolver`` is the one owner of the thought-channel ids. It takes
an ``enc(text) -> ids`` callable that encodes without special tokens."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Callable

from .reads import Slot
from .schema import SchemaError

# Answer template shape: (join between questions, what precedes the label,
# reply instruction). "indexed" costs three tokens a question against four
# or five for "lines", and keeps a large schema in one read.
FORMATS = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}

# The empty thought block the chat template leaves to the model.
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"
THOUGHT_OPEN_TEXT = "<|channel>thought\n"
THOUGHT_CLOSE_TEXT = "<channel|>"


def system_text(schema, chunked: bool = False) -> str:
    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}: {str(desc).strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += "\n" + FORMATS[schema.get("format", "lines")][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line "
            "that is present."
        )
    return s


def answer_text(qs, labels, fmt: str = "lines") -> str:
    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i] for q, i in zip(qs, labels)
    )


class TemplateResolver:
    """Tokenizes answer templates and finds each question's slot.

    ``canvas_len`` is the served canvas. Widths are multiples of
    ``canvas_step`` up to it. Resolved templates are kept in an LRU of
    ``cache_size`` entries."""

    def __init__(
        self,
        enc: Callable[[str], list[int]],
        canvas_len: int,
        canvas_step: int = 16,
        cache_size: int = 256,
    ):
        self.enc = enc
        self.canvas_len = int(canvas_len)
        self.canvas_step = int(canvas_step)
        self.thought_open = list(enc(THOUGHT_OPEN_TEXT))
        self.thought_close = list(enc(THOUGHT_CLOSE_TEXT))
        self.scaffold = list(enc(SCAFFOLD_TEXT))
        if self.thought_open + self.thought_close != self.scaffold:
            raise RuntimeError(
                "the tokenizer does not split the thought tags apart; "
                "structured reads need a DiffusionGemma tokenizer"
            )
        if len(self.thought_close) != 1:
            raise RuntimeError(
                "the thought close tag is not a single token; structured "
                "reads need a DiffusionGemma tokenizer"
            )
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()

    def canvas_width(self, template) -> int:
        """Smallest multiple of the canvas step that holds the template and
        the turn close."""
        need = len(template) + 1
        step = self.canvas_step
        return min(self.canvas_len, -(-need // step) * step)

    def rows_for(self, qs, fmt: str) -> int:
        """Canvas rows a group's plain read needs: the scaffold, the answer
        lines and the turn close."""
        return (
            len(self.scaffold)
            + len(self.enc(answer_text(qs, [0] * len(qs), fmt)))
            + 1
        )

    def resolve(self, qs, head, lead, fmt) -> tuple[tuple[int, ...], tuple[Slot, ...]]:
        """Tokenize the answer template and find each question's slot. Every
        label must change exactly one token, at the same position for all of
        a question's labels, or this raises SchemaError. ``head`` is the
        token run the canvas starts with. ``lead`` is the text before the
        first answer."""
        base_labels = [0] * len(qs)
        base = list(head) + self.enc(lead + answer_text(qs, base_labels, fmt))
        if len(base) + 1 > self.canvas_len:
            raise SchemaError(
                f"answer template is {len(base)} tokens; the canvas holds "
                f"{self.canvas_len - 1}"
            )
        slots = []
        for qi, q in enumerate(qs):
            pos = None
            ids = [0] * len(q["labels"])
            for li in range(1, len(q["labels"])):
                labels = list(base_labels)
                labels[qi] = li
                e = list(head) + self.enc(lead + answer_text(qs, labels, fmt))
                if len(e) != len(base):
                    raise SchemaError(
                        f"question {q['id']!r}: label {q['labels'][li]!r} is not a "
                        "single token"
                    )
                diffs = [i for i in range(len(e)) if e[i] != base[i]]
                if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                    raise SchemaError(
                        f"question {q['id']!r}: labels do not share one template slot"
                    )
                pos = diffs[0]
                ids[li] = e[pos]
            assert pos is not None  # every question has two or more labels
            ids[0] = base[pos]
            if len(set(ids)) != len(ids):
                raise SchemaError(
                    f"question {q['id']!r}: two labels tokenize to the same id"
                )
            slots.append(Slot(pos=pos, label_ids=tuple(ids)))
        return tuple(base), tuple(slots)

    def template_for(self, schema, head, lead):
        fmt = schema.get("format", "lines")
        key = json.dumps(
            [list(head), lead, fmt]
            + [(q["id"], q["labels"]) for q in schema["questions"]]
        )
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        value = self.resolve(schema["questions"], head, lead, fmt)
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return value
