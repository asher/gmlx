# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM examples/features/structured_diffusion/structured_server.py
# at ab3de6edf2 and modified for gmlx (Apache-2.0; see
# licenses/vllm-LICENSE).
"""One structured decision: stages, chunks, samples and the aggregation.

The logic matches the vLLM proxy for a text state. The proxy's HTTP reads
become calls on a ``ReadEngine``: one batched read per group, with every
sample in it, over a prompt that is prefilled once and extended for later
stages when its tokens allow."""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Protocol

from .reads import Cancelled, ReadRequest, ReadResult, pinned_positions
from .schema import schedule
from .template import FORMATS, TemplateResolver, answer_text, system_text

SAMPLE_SEED_STRIDE = 7919
GROUP_SEED_STRIDE = 104729


class PromptCacheLike(Protocol):
    ids: list[int]

    def extend(self, ids: list[int]) -> None: ...


class ReadEngine(Protocol):
    def prefill(self, ids: list[int]) -> PromptCacheLike: ...

    def read(self, prompt: PromptCacheLike, req: ReadRequest) -> ReadResult: ...

    def think(
        self, prompt_ids: list[int], budget: int, *, stop_id: int, canvas_width: int
    ) -> tuple[list[int], dict]: ...


def slot_distribution(top: dict, label_ids) -> dict:
    """Label probabilities at one slot from the returned logprobs: every
    label's own value plus the argmax token. The logprobs are at
    temperature 1, so the label softmax uses them directly. The entropy is
    over the returned set."""
    floor = min(top.values()) - 5.0
    lp_t = [top.get(i, floor) for i in label_ids]
    mx = max(lp_t)
    ex = [math.exp(x - mx) for x in lp_t]
    probs = [e / sum(ex) for e in ex]
    top_p = [math.exp(v) for v in top.values()]
    return {
        "probs": probs,
        "label_mass": sum(math.exp(x) for x in lp_t),
        "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
        "argmax_is_label": max(top, key=top.get) in label_ids,
    }


def answer_name(q, a):
    """The answer as the name ask_if compares against: yes or no, an option
    name, or a level name."""
    if a is None:
        return None
    return a["label"] if q["type"] == "noul" else a.get("choice", a.get("level"))


def chunk_groups(schema, qs, resolver: TemplateResolver):
    """``qs`` split, in order, into the fewest groups whose answer templates
    fit ``chunk_rows`` (the canvas by default). A question marked alone gets
    its own group."""
    limit = schema.get("chunk_rows") or resolver.canvas_len
    fmt = schema.get("format", "lines")
    groups, group = [], []
    for q in qs:
        if q["alone"]:
            if group:
                groups.append(group)
                group = []
            groups.append([q])
            continue
        trial = group + [q]
        if resolver.rows_for(trial, fmt) > limit and group:
            groups.append(group)
            group = [q]
        else:
            group = trial
    if group:
        groups.append(group)
    return groups


class _Decision:
    """The per-decision state: the engine, the tokenizer callables, and the
    one prompt cache kept between groups."""

    def __init__(self, engine, resolver, chat_ids, constrained, canvas_len,
                 decode, should_stop):
        self.engine = engine
        self.resolver = resolver
        self.chat_ids = chat_ids
        self.constrained = constrained
        self.canvas_len = canvas_len
        self.decode = decode
        self.should_stop = should_stop
        self.cache: Optional[PromptCacheLike] = None

    def check(self):
        if self.should_stop is not None and self.should_stop():
            raise Cancelled("decision cancelled")

    def prompt_cache(self, target: list[int]) -> PromptCacheLike:
        """A cache holding ``target``: the current one extended when its ids
        are a prefix of the target, else a fresh prefill."""
        cur = self.cache
        if cur is not None:
            n = len(cur.ids)
            if n <= len(target) and list(cur.ids) == list(target[:n]):
                if n < len(target):
                    cur.extend(list(target[n:]))
                return cur
        self.cache = None
        self.check()
        self.cache = self.engine.prefill(list(target))
        return self.cache

    def think(self, sys_text: str, budget: int):
        """A read prefix that ends with a thought the model wrote: the chat
        prompt with thinking on, the open tag, up to ``budget`` generated
        tokens, the close tag."""
        self.check()
        prompt = list(self.chat_ids(sys_text, True)) + self.resolver.thought_open
        started = time.time()
        ids, info = self.engine.think(
            prompt, budget, stop_id=self.resolver.thought_close[0],
            canvas_width=self.canvas_len)
        ms = info.get("ms")
        info = {
            "tokens": len(ids),
            "closed": bool(info.get("closed")),
            "ms": ms if ms is not None else (time.time() - started) * 1e3,
            "text": self.decode(ids),
        }
        return prompt + list(ids) + self.resolver.thought_close, info

    def read_many(self, schema, template, slots, prompt_ids, seed, n):
        self.check()
        cache = self.prompt_cache(prompt_ids)
        width = self.resolver.canvas_width(template)
        steps = schema["steps"]
        req = ReadRequest(
            template=tuple(template),
            slots=tuple(slots),
            width=width,
            seeds=tuple(seed + k * SAMPLE_SEED_STRIDE for k in range(n)),
            steps=steps,
            constrained=self.constrained,
            pinned=bool(pinned_positions(slots, width, steps)),
        )
        result = self.engine.read(cache, req)
        return [
            [slot_distribution(sr.top, s.label_ids)
             for sr, s in zip(sample.slots, slots)]
            for sample in result.samples
        ]

    def group(self, schema, sys_text, seed, prefix=None, lead=""):
        started = time.time()
        thought = None
        head = self.resolver.scaffold if prefix is None else []
        if prefix is None and schema["think"]:
            prefix, thought = self.think(sys_text, schema["think"])
            head = []
        template, slots = self.resolver.template_for(schema, head, lead)
        prompt_ids = (list(prefix) if prefix is not None
                      else list(self.chat_ids(sys_text, False)))
        policy = schema["policy"]
        if policy["mode"] == "fixed":
            reads = self.read_many(schema, template, slots, prompt_ids, seed,
                                   policy["n"])
            extended = None
            first_entropy = None
        else:
            reads = self.read_many(schema, template, slots, prompt_ids, seed, 1)
            first_entropy = {
                q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])
            }
            extended = (
                max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
            )
            if extended:
                reads += self.read_many(schema, template, slots, prompt_ids,
                                        seed + 1, policy["max"] - 1)
        prompt_tokens = len(prompt_ids) or None
        elapsed_ms = (time.time() - started) * 1e3

        answers = {}
        diag_q = {}
        n = len(reads)
        for qi, q in enumerate(schema["questions"]):
            per = [r[qi]["probs"] for r in reads]
            mean = [sum(p[i] for p in per) / n for i in range(len(q["labels"]))]
            top = max(range(len(mean)), key=lambda i: mean[i])
            a = {
                "type": q["type"],
                "label": q["labels"][top],
                "confidence": mean[top],
                "probabilities": {c[0]: m for c, m in zip(q["choices"], mean)},
            }
            if q["type"] == "noul":
                a["noul"] = mean[0]
            elif q["type"] == "choice":
                a["choice"] = q["choices"][top][0]
            else:
                a["score"] = sum((i + 1) * m for i, m in enumerate(mean))
                a["level"] = q["choices"][top][0]
            if n > 1:
                var = sum((p[top] - mean[top]) ** 2 for p in per) / (n - 1)
                a["stderr"] = (var / n) ** 0.5
                a["agreement"] = (
                    sum(1 for p in per
                        if max(range(len(p)), key=lambda i: p[i]) == top) / n
                )
            answers[q["id"]] = a
            diag_q[q["id"]] = {
                "pos": slots[qi].pos,
                "entropy": [r[qi]["entropy"] for r in reads],
                "label_mass": reads[0][qi]["label_mass"],
                "argmax_is_label": reads[0][qi]["argmax_is_label"],
            }
        tops = [
            {
                q["id"]: [
                    q["labels"][
                        max(range(len(r[qi]["probs"])),
                            key=lambda i: r[qi]["probs"][i])
                    ],
                    max(r[qi]["probs"]),
                    r[qi]["entropy"],
                ]
                for qi, q in enumerate(schema["questions"])
            }
            for r in reads
        ]
        return {
            "answers": answers,
            "diagnostics": {
                "steps": schema["steps"],
                "samples": {
                    "n": n,
                    "tops": tops,
                    "policy": dict(
                        policy, extended=extended, first_read_entropy=first_entropy
                    ),
                },
                "timing": {"total_ms": elapsed_ms, "reads": n},
                "thought": thought,
                "prompt_tokens": prompt_tokens,
                "questions": diag_q,
                "engine": "gmlx",
            },
        }, len(template) + 1 + (thought["tokens"] if thought else 0)


def decide_once(
    schema,
    *,
    engine: ReadEngine,
    resolver: TemplateResolver,
    chat_ids: Callable[[str, bool], list[int]],
    seed: int,
    constrained: bool,
    canvas_len: int,
    decode: Callable[[list[int]], str],
    should_stop: Optional[Callable[[], bool]] = None,
):
    """One decision run, as the vLLM example's ``decide`` for a text state.
    ``chat_ids(system_text, thinking)`` renders the chat prompt as token ids.
    Returns the body (answers and diagnostics) and the completion-token
    count.

    Questions run in stages by their dependencies. A stage is one joint
    read, chunked by the canvas, with a question marked alone in its own
    read. Later stages continue the earlier answers as a prefilled prompt. A
    question whose ask_if condition failed is skipped and its answer is
    null."""
    d = _Decision(engine, resolver, chat_ids, constrained, canvas_len, decode,
                  should_stop)
    started = time.time()
    qs = [
        q
        for q in schema["questions"]
        if not schema.get("ask") or q["id"] in schema["ask"]
    ]
    levels = schedule(qs)
    fmt = schema["format"]
    join = FORMATS[fmt][0]
    # More than one read in sequence needs the full question list in every
    # prompt, so that later reads continue one answer.
    chained = len(levels) > 1 or schema["sequential"]
    sys_full = system_text(schema)
    base_ids, thought = None, None
    if chained:
        if schema["think"]:
            base_ids, thought = d.think(sys_full, schema["think"])
        else:
            base_ids = list(chat_ids(sys_full, False)) + resolver.scaffold
    shared = schema["chunk_prompt"] == "shared"
    answered, lines = {}, []
    parts, stages, chunks, skipped = [], [], [], {}
    by_id = {q["id"]: q for q in schema["questions"]}

    def run(group, k, conditioned):
        sub = dict(schema, questions=group)
        # The thought is written once: in the prefix of a chained decision,
        # or in the first read of anything else.
        sub["think"] = 0 if chained or conditioned else schema["think"]
        if conditioned:
            assert base_ids is not None
            prefix, lead, sys_text = (
                base_ids + resolver.enc(join.join(lines)), join, sys_full)
        elif chained:
            prefix, lead, sys_text = (base_ids if thought else None), "", sys_full
        else:
            prefix, lead, sys_text = (
                None,
                "",
                system_text(schema, chunked=True) if shared else system_text(sub),
            )
        return d.group(sub, sys_text, seed + GROUP_SEED_STRIDE * k, prefix, lead)

    def absorb(group, body, rows):
        parts.append((body, rows))
        chunks.append([q["id"] for q in group])
        answered.update(body["answers"])
        lines.append(
            answer_text(
                group,
                [q["labels"].index(body["answers"][q["id"]]["label"]) for q in group],
                fmt,
            )
        )

    k = 0
    for level in levels:
        asked = []
        for q in level:
            failed = next(
                (
                    (dep, vals)
                    for dep, vals in q["ask_if"].items()
                    if answer_name(by_id[dep], answered.get(dep)) not in vals
                ),
                None,
            )
            if failed:
                answered[q["id"]] = None
                skipped[q["id"]] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": failed[1],
                }
                continue
            asked.append(q)
        if not asked:
            continue
        stages.append([q["id"] for q in asked])
        groups = chunk_groups(schema, asked, resolver)
        conditioned = bool(lines)
        if schema["sequential"] or len(groups) == 1:
            for group in groups:
                body, rows = run(
                    group, k, conditioned or (schema["sequential"] and bool(lines))
                )
                absorb(group, body, rows)
                k += 1
        else:
            # The proxy reads these chunks in parallel. They share one
            # conditioning, so reading them in order gives the same answers.
            results = [run(g, k + i, conditioned) for i, g in enumerate(groups)]
            for group, (body, rows) in zip(groups, results):
                absorb(group, body, rows)
            k += len(groups)

    answers = {q["id"]: answered.get(q["id"]) for q in qs}
    diag_q = {}
    for body, _ in parts:
        diag_q.update(body["diagnostics"]["questions"])
    # A thought written here (a chained decision) is outside every group's
    # row count. One written inside a group is already counted there.
    extra_rows = thought["tokens"] if thought else 0
    if thought is None:
        thoughts = [b["diagnostics"].get("thought") for b, _ in parts]
        thought = (
            thoughts[0] if len(parts) == 1 else ([t for t in thoughts if t] or None)
        )
    one = len(parts) == 1 and not skipped
    diagnostics = {
        "steps": schema["steps"],
        "stages": stages,
        "skipped": skipped,
        "chunks": chunks,
        "chunk_prompt": "full" if chained else schema["chunk_prompt"],
        "sequential": schema["sequential"],
        "conditioning": (
            None if len(stages) <= 1 and not schema["sequential"] else "prefill"
        ),
        "thought": thought,
        "samples": (
            parts[0][0]["diagnostics"]["samples"]
            if one
            else {
                "n": [b["diagnostics"]["samples"]["n"] for b, _ in parts],
                "tops": [b["diagnostics"]["samples"]["tops"] for b, _ in parts],
                "policy": [b["diagnostics"]["samples"]["policy"] for b, _ in parts],
            }
        ),
        "timing": {
            "total_ms": (time.time() - started) * 1e3,
            "reads": sum(b["diagnostics"]["timing"]["reads"] for b, _ in parts),
        },
        "prompt_tokens": max(
            (b["diagnostics"].get("prompt_tokens") or 0) for b, _ in parts
        )
        or None,
        "questions": diag_q,
        "engine": "gmlx",
    }
    return {"answers": answers, "diagnostics": diagnostics}, sum(
        rows for _, rows in parts
    ) + extra_rows
