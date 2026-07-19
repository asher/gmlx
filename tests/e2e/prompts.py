"""The prompt suite the harness fires at every server config.

Each builder returns a :class:`PromptInstance`. The suite deliberately spans
*depth*: short factual turns, instruction-following, a long planted-fact
("needle") context that stresses prefill + KV recall, and a long generation that
stresses decode at depth - because degeneration (looping, drift, KV corruption)
usually only shows up with a big context or a long answer, not on a one-liner.

Pure stdlib; capabilities tag VLM prompts so text-only scenarios skip them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptInstance:
    key: str
    messages: list                      # OpenAI chat messages
    max_tokens: int = 256
    kind: str = "short"                 # short|instruct|system|long_ctx|long_gen|multiturn|vlm
    needs: tuple = ()                   # capabilities, e.g. ("vlm",)
    # optional anchored expectations (the judge is the primary semantic gate)
    anchors: dict = field(default_factory=dict)   # {"substrs": [...], "mode": "any|all"}
    judge: bool = True                  # send to the LLM judge in the judging phase
    # how to summarize the request to the judge (keeps long needles out of the judge)
    judge_request: Optional[str] = None

    def request_summary(self) -> str:
        if self.judge_request is not None:
            return self.judge_request
        last_user = next((m["content"] for m in reversed(self.messages)
                          if m["role"] == "user"), "")
        if isinstance(last_user, list):     # multimodal content parts
            last_user = " ".join(p.get("text", "") for p in last_user
                                 if isinstance(p, dict))
        return last_user[:600]


def _u(text):
    return {"role": "user", "content": text}


def _s(text):
    return {"role": "system", "content": text}


def _a(text):
    return {"role": "assistant", "content": text}


# builders
# Deterministic single-fact prompts carry a strong anchor, which IS the correctness
# oracle, so they skip the LLM judge: an arithmetic/recall answer the anchor already
# confirmed has nothing for a semantic judge to add but a false-negative (gemma-4-12B
# once scored the correct answer "42" a 1, "wrong math"). The judge earns its keep on
# the open-ended prompts (instruct format, count completeness, long_gen, system, vlm).
def p_capital() -> PromptInstance:
    return PromptInstance(
        "capital", [_u("What is the capital of France? Answer with just the city name.")],
        max_tokens=24, kind="short", judge=False,
        anchors={"substrs": ["paris"], "mode": "any"})


def p_math() -> PromptInstance:
    return PromptInstance(
        "math", [_u("What is 17 + 25? Reply with only the number.")],
        max_tokens=24, kind="short", judge=False,
        anchors={"substrs": ["42"], "mode": "any"})


def p_instruct() -> PromptInstance:
    return PromptInstance(
        "instruct",
        [_u("List exactly three primary colors, one per line, no other text.")],
        max_tokens=64, kind="instruct",
        anchors={"substrs": ["red", "blue", "yellow"], "mode": "all"})


def p_system_uppercase() -> PromptInstance:
    return PromptInstance(
        "system_uppercase",
        [_s("You MUST write your entire reply in ALL UPPERCASE letters."),
         _u("Tell me you are ready to help, in one short sentence.")],
        max_tokens=48, kind="system",
        judge_request="(system: reply in ALL UPPERCASE) say you are ready to help")


def p_multiturn() -> PromptInstance:
    return PromptInstance(
        "multiturn",
        [_u("My favorite animal is the octopus. Remember that."),
         _a("Got it - your favorite animal is the octopus."),
         _u("What did I say my favorite animal was? Answer in one word.")],
        max_tokens=24, kind="multiturn", judge=False,
        anchors={"substrs": ["octopus"], "mode": "any"})


def p_count() -> PromptInstance:
    return PromptInstance(
        "count",
        [_u("Count from 1 to 40, separated by single spaces. Output only the numbers.")],
        max_tokens=160, kind="instruct",
        anchors={"substrs": ["1", "20", "40"], "mode": "all"})


def p_long_gen() -> PromptInstance:
    return PromptInstance(
        "long_gen",
        [_u("Write a clear, well-structured 6-paragraph explanation of how a "
            "four-stroke internal combustion engine works, covering intake, "
            "compression, power, and exhaust strokes, plus valves and ignition.")],
        max_tokens=900, kind="long_gen",
        judge_request="write a detailed 6-paragraph explanation of a four-stroke engine")


def p_long_ctx_needle(passcode: str = "BLUEMANGO47",
                      *, filler_paras: int = 26) -> PromptInstance:
    """A long context with a single planted fact, then a recall question. Exercises
    deep prefill + KV recall - the scenario that most reliably exposes a quantized-KV
    or long-context regression. The passcode is the anchor and should be unique per
    call where cache cross-talk must be avoided."""
    filler = (
        "The logistics review covered warehouse throughput, seasonal demand "
        "curves, carrier SLAs, and the migration of the returns pipeline to the "
        "new regional hub. Stakeholders debated buffer stock levels and the "
        "trade-offs between expedited freight and inventory carrying cost.")
    body = []
    for i in range(filler_paras):
        body.append(f"Section {i + 1}. {filler}")
        if i == filler_paras // 2:
            body.append(f"IMPORTANT OPERATIONAL NOTE: the secret passcode is "
                        f"{passcode}. Commit it to memory; it will be asked for.")
    doc = "\n\n".join(body)
    return PromptInstance(
        "long_ctx_needle",
        [_u(doc + "\n\nQuestion: What is the secret passcode mentioned in the "
            "operational note? Reply with only the passcode.")],
        max_tokens=32, kind="long_ctx",
        anchors={"substrs": [passcode], "mode": "any"},
        judge=False,                    # purely an anchored recall test
        judge_request="(long planted-fact context) recall the secret passcode")


def p_vlm_describe() -> PromptInstance:
    # The image is attached by the client (it needs the local path/data-uri); the
    # text part is fixed. Marked vlm so text-only scenarios skip it.
    return PromptInstance(
        "vlm_describe",
        [_u("Describe what is in this image in one or two sentences.")],
        max_tokens=128, kind="vlm", needs=("vlm",),
        judge_request="describe the attached image")


# named bundles the scenarios pick from
def core_suite() -> list:
    return [p_capital(), p_math(), p_instruct(), p_multiturn(), p_count(),
            p_long_gen()]


def quick_suite() -> list:
    return [p_capital(), p_instruct(), p_long_gen()]


def depth_suite() -> list:
    """Depth-stressing prompts for KV-quant / cache / long-context scenarios."""
    return [p_long_ctx_needle(), p_long_gen()]


def system_suite() -> list:
    return [p_system_uppercase(), p_capital()]
