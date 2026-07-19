"""Built-in per-family sampling profiles ("intents").

Model vendors publish recommended sampling for each family -- and they disagree
(Gemma wants temperature 1.0 and degrades at low temperature; Qwen3.6 publishes
three distinct operating points; gpt-oss wants temperature 1.0 / top_p 1.0 plus
a reasoning-effort switch). This module is the single table of those
recommendations, keyed by GGUF architecture, and the resolver that turns
(family, intent) into profile groups.

How it reaches a request (see ``config.resolve_model``):

* Every model's family **base** group is the lowest sampling layer -- beneath
  ``server.defaults.profile``, so anything the user configures still wins.
* The **intents** (``coding``, ``creative``, ``instruct``, ``reasoning-*``) are
  addressable like user profiles: ``id@coding``, a request ``profile`` field,
  or ``run/chat --profile coding``. A family without a card-specific value for
  an intent resolves it to its own base (per Gemma's guidance, ``@coding`` on a
  Gemma model deliberately stays at temperature 1.0).
* A user profile with the same name shadows the built-in; ``extends: coding``
  composes over it; a per-model ``profiles:`` block tweaks it for one model.
* ``server: {family_defaults: false}`` switches all of this off.

Values are copied from the primary model cards (cited per family, retrieved
2026-07). Only card-backed numbers appear here; nothing is invented beyond the
``default`` family, which carries the historic scaffold defaults for unknown
architectures. Families never set ``max_tokens`` -- that stays a user concern.

Pure-Python and import-free: ``config.py`` imports this module, never the
reverse.
"""

from __future__ import annotations

import re

# Family table. Per family: the GGUF arches it covers (optionally refined by a
# general.name regex when one arch spans families), the base group, and the
# intent deltas (merged over base when that intent is addressed).
# Group shape mirrors a profile: {"sampling": {...}, "chat_template_kwargs": {...}}.
FAMILIES: dict[str, dict] = {
    # https://huggingface.co/Qwen/Qwen3.6-27B (+ 35B-A3B), 2026-07.
    # Base = "thinking mode, general tasks" (t=1.0/top_p=0.95/top_k=20).
    # presence_penalty differs between the 27B (0.0) and 35B-A3B (1.5) cards,
    # so the base omits it; the instruct point carries the shared 1.5.
    # Covers Qwen3.5 (same thinking/coding points) and Qwen3-Next.
    "qwen3.6": {
        "label": "Qwen3.5 / Qwen3.6",
        "arches": ("qwen35", "qwen35moe", "qwen3next"),
        "base": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                              "min_p": 0.0}},
        "intents": {
            # "thinking mode, precise coding tasks (e.g. WebDev)"
            "coding": {"sampling": {"temperature": 0.6, "top_p": 0.95,
                                    "top_k": 20, "min_p": 0.0}},
            # "instruct (or non-thinking) mode"
            "instruct": {
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "min_p": 0.0, "presence_penalty": 1.5},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    },
    # https://huggingface.co/Qwen/Qwen3-30B-A3B (thinking/non-thinking), 2026-07.
    "qwen3": {
        "label": "Qwen3",
        "arches": ("qwen3", "qwen3moe", "qwen3vlmoe"),
        "base": {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                              "min_p": 0.0}},
        "intents": {
            "instruct": {
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "min_p": 0.0},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    },
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct generation_config, 2026-07.
    "qwen2.5": {
        "label": "Qwen2 / Qwen2.5",
        "arches": ("qwen2", "qwen2moe"),
        "base": {"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                              "repetition_penalty": 1.05}},
        "intents": {},
    },
    # https://ai.google.dev/gemma/docs/core/model_card_4, 2026-07. One point
    # across use cases; the card warns low temperature degrades output, so no
    # coding/instruct deltas exist on purpose (they resolve to base).
    "gemma": {
        "label": "Gemma",
        "arches": ("gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
                   "diffusion-gemma"),
        "base": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}},
        "intents": {},
    },
    # https://github.com/openai/gpt-oss + gpt-oss-120b card, 2026-07.
    # top_k deliberately unset (the card says disable it). reasoning_effort is
    # a chat-template variable in the GGUF-embedded harmony template.
    "gpt-oss": {
        "label": "gpt-oss",
        "arches": ("gpt-oss",),
        "base": {"sampling": {"temperature": 1.0, "top_p": 1.0}},
        "intents": {
            "reasoning-low": {"chat_template_kwargs": {"reasoning_effort": "low"}},
            "reasoning-medium": {"chat_template_kwargs": {"reasoning_effort": "medium"}},
            "reasoning-high": {"chat_template_kwargs": {"reasoning_effort": "high"}},
        },
    },
    # https://huggingface.co/zai-org/GLM-5.2 (+ GLM-4.x cards), 2026-07.
    "glm": {
        "label": "GLM",
        "arches": ("glm4", "glm4moe", "glm-dsa"),
        "base": {"sampling": {"temperature": 1.0, "top_p": 0.95}},
        "intents": {},
    },
    # DeepSeek-R1 / V3 usage recommendations, 2026-07 (llama.cpp 'deepseek2'
    # also covers GLM MLA conversions, which tolerate these; 'deepseek4' is
    # V4 Flash, same recommended sampling per its model card).
    "deepseek": {
        "label": "DeepSeek",
        "arches": ("deepseek2", "deepseek4"),
        "base": {"sampling": {"temperature": 0.6, "top_p": 0.95}},
        "intents": {},
    },
    # https://huggingface.co/MiniMaxAI/MiniMax-M2.7, 2026-07.
    "minimax": {
        "label": "MiniMax",
        "arches": ("minimax-m2", "minimax-m3"),
        "base": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40}},
        "intents": {},
    },
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, 2026-07.
    "nemotron": {
        "label": "Nemotron",
        "arches": ("nemotron_h_moe",),
        "base": {"sampling": {"temperature": 1.0, "top_p": 0.95}},
        "intents": {},
    },
    # https://huggingface.co/tencent/Hunyuan-A13B-Instruct, 2026-07.
    "hunyuan": {
        "label": "Hunyuan",
        "arches": ("hunyuan-moe",),
        "base": {"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                              "repetition_penalty": 1.05}},
        "intents": {},
    },
    # https://huggingface.co/tencent/Hy3 generation_config (t=0.9, top_p/top_k
    # off), 2026-07. The chat template's reasoning_effort takes
    # no_think/low/high (defaults to no_think; other values raise in-template).
    # The thinking tokens carry Hy3's ':opensource' tag suffix so the server's
    # open-think detection, budget criteria, and stream splitter see the
    # model's real markers instead of the '<think>' default.
    "hy3": {
        "label": "Hy3",
        "arches": ("hy_v3",),
        "base": {"sampling": {
            "temperature": 0.9,
            "thinking_start_token": "<think:opensource>",
            "thinking_end_token": "</think:opensource>",
        }},
        "intents": {
            "reasoning-low": {"chat_template_kwargs": {"reasoning_effort": "low"}},
            "reasoning-high": {"chat_template_kwargs": {"reasoning_effort": "high"}},
        },
    },
    # Llama 3.x generation_config (t=0.6/top_p=0.9); SmolLM3 card matches
    # closely enough to share.
    "llama": {
        "label": "Llama",
        "arches": ("llama", "smollm3"),
        "base": {"sampling": {"temperature": 0.6, "top_p": 0.9}},
        "intents": {},
    },
    # Mistral-Small-3.x cards recommend a very low temperature, 2026-07.
    "mistral": {
        "label": "Mistral",
        "arches": ("mistral3",),
        "base": {"sampling": {"temperature": 0.15}},
        "intents": {},
    },
    # Unknown architectures: the historic scaffold defaults, plus generic
    # intent deltas so @coding/@creative still mean something.
    "default": {
        "label": "generic",
        "arches": (),
        "base": {"sampling": {"temperature": 0.7, "top_p": 0.95}},
        "intents": {
            "coding": {"sampling": {"temperature": 0.3}},
            "creative": {"sampling": {"temperature": 1.0, "min_p": 0.05}},
            "instruct": {},
        },
    },
}

# Every intent name any family defines -- the set that is addressable as
# @intent on any model (a family without its own value resolves it to base).
BUILTIN_INTENTS: frozenset = frozenset(
    name for fam in FAMILIES.values() for name in fam["intents"]
)

_ARCH_TO_FAMILY: dict[str, str] = {
    arch: key for key, fam in FAMILIES.items() for arch in fam["arches"]
}

# Optional general.name refinement for arches that span sampling families.
# Checked before _ARCH_TO_FAMILY; first hit wins. None currently needed --
# the hook exists so a future split (e.g. one arch, two cards) is one line.
_NAME_REFINEMENTS: tuple = (
    # (arch, compiled-regex-on-general.name, family-key)
)


def detect_family(arch: str | None, name: str | None = None) -> str:
    """The family key for a GGUF's ``general.architecture`` (optionally refined
    by ``general.name``). Unknown/None arch -> ``"default"``."""
    if arch:
        for a, rx, key in _NAME_REFINEMENTS:
            if a == arch and name and re.search(rx, name):
                return key
        fam = _ARCH_TO_FAMILY.get(arch)
        if fam:
            return fam
    return "default"


def builtin_intents() -> frozenset:
    """All addressable built-in intent names."""
    return BUILTIN_INTENTS


def family_base(family: str | None) -> dict:
    """The family's base group (``{"sampling": ...}``); the ``default``
    family's when unknown/None."""
    fam = FAMILIES.get(family or "default", FAMILIES["default"])
    return fam["base"]


def family_intents(family: str | None) -> dict:
    """Every built-in intent resolved for ``family``: the family's own delta
    where the card defines one, else an empty delta (-> base). Deltas are
    merged over base by the caller."""
    fam = FAMILIES.get(family or "default", FAMILIES["default"])
    return {name: fam["intents"].get(name, {}) for name in BUILTIN_INTENTS}


def groups_for(family: str | None, intent: str | None = None) -> dict:
    """Merged ``{"sampling": ..., "chat_template_kwargs": ...}`` for a family
    (base, or base + intent delta). Unknown intent -> base unchanged."""
    base = family_base(family)
    delta = family_intents(family).get(intent or "", {})
    out = {
        "sampling": {**base.get("sampling", {}), **delta.get("sampling", {})},
        "chat_template_kwargs": {**base.get("chat_template_kwargs", {}),
                                 **delta.get("chat_template_kwargs", {})},
    }
    return {k: v for k, v in out.items() if v}


def describe() -> list[dict]:
    """The full table, one row per family: key, label, covered arches, base
    group, and per-intent resolved groups. Drives ``gmlx profiles``, the docs
    table, and the docs-drift test."""
    rows = []
    for key, fam in FAMILIES.items():
        rows.append({
            "family": key,
            "label": fam["label"],
            "arches": list(fam["arches"]),
            "base": fam["base"],
            "intents": {name: groups_for(key, name)
                        for name in sorted(BUILTIN_INTENTS)},
        })
    return rows
