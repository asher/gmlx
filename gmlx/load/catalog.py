"""The chat models ``gmlx pull`` recommends, and the plan a pull of one makes.

``MODELS`` names each entry: its repository at a pinned revision, the
quality tiers (``fastest``, ``balanced``, ``high``, ``xhigh``) and compact
quants it offers, its vision tower and drafter, and the figures the
refresh script recorded (sizes, lookup-table bytes, KV costs, the trained
context). ``RECOMMENDED`` lists the entries to lead with per memory
class. The editorial rows are ``_SEED`` here; the measured facts are
``catalog_facts.FACTS``, printed whole by ``scripts/refresh_pull_catalog
.py`` and pasted there.

The helpers take a ``Box`` (this Mac's RAM, working set, resident budget,
ceiling and prefill cap) and answer, from recorded figures alone: a
rung's state (in memory, from the SSD, too big), the default plan (rung
and add-ons together), how much context a plan leaves room for, the
groups the picker shows, and the ``gmlx pull`` command that reproduces a
plan. Module-level imports are stdlib only; the helpers import gmlx's
memory rules when they run.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

GiB = 1 << 30
K = 1024

# The memory classes a recommendation is made for, in GiB of RAM. A Mac
# takes the largest class at or under its RAM.
MEMORY_CLASSES = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512)
TIERS = ("fastest", "balanced", "high", "xhigh")

RESIDENT = "resident"       # the file fits in memory
STREAMS = "streams"         # a MoE over the bar that runs from the SSD here
OVER = "over"               # too big for this Mac

LICENSE_WORDS = {"apache-2.0": "Apache 2.0", "mit": "MIT"}


@dataclass(frozen=True)
class Rung:
    """One file of an entry: a quality tier or a compact quant."""
    quant: str
    bytes: int                       # summed shard bytes, the download
    path: str                        # in the repo
    tier: str = ""                   # a TIERS name, or "" for a compact quant
    text_bytes: int = 0              # text tensors without a native head's block; 0 = bytes
    table_bytes: int = 0             # lookup tables a load reads from the file
    streams_from_gib: int | None = None   # MoE only; None = never streams
    head: bool = False               # this file carries a native speculation head


@dataclass(frozen=True)
class Drafter:
    kind: str                        # as discovery names it: dflash2, dflash or the arch
    repo: str
    revision: str
    file: str
    bytes: int
    cross_base: bool = False         # trained for another checkpoint than the entry's


@dataclass(frozen=True)
class KvCost:
    """The entry's full-precision cache cost per token: rows of
    ``(kind, n, bytes_per_token, count)`` with kind ``full`` (grows with
    the context), ``window`` (a sliding window of n rows), ``rows`` (n
    rows charged at the first token) or ``step`` (slabs of n tokens)."""
    costs: tuple


@dataclass(frozen=True)
class Model:
    name: str
    title: str
    repo: str
    revision: str
    rungs: tuple                     # ascending by bytes
    preferred: str                   # a tier, or a quant for an entry with no tiers
    params: str
    blurb: str                       # one sentence: what it is good at
    speed: str                       # one sentence: how quick it feels
    license: str
    use: str = ""                    # a few words: what it is good for
    license_url: str = ""
    license_note: str = ""           # the license's name when it is not a common one
    license_detail: str = ""         # one sentence on a limit worth knowing before the pull
    aliases: tuple = ()
    arch: str = ""
    moe: bool = False
    native_mtp: bool = False         # every rung carries a head
    mmproj: str | None = None
    mmproj_bytes: int = 0
    drafter: Drafter | None = None
    ignored_files: tuple = ()        # repo companions the seed leaves out on purpose
    kv_cost: KvCost | None = None
    heads: int = 0
    trained_ctx: int = 0
    preferred_at: tuple = ()         # (memory class, tier) pairs, over preferred on that class


@dataclass(frozen=True)
class Box:
    """One Mac: RAM, Metal working set, resident budget, the ceiling on
    tracked bytes and the prefill transient cap, all in bytes."""
    ram: int
    ws: int
    budget: int
    ceiling: int
    cap: int


@dataclass(frozen=True)
class PullPlan:
    model: Model
    rung: Rung
    mmproj: bool
    drafter: bool
    streaming: bool


@dataclass(frozen=True)
class ImageEntry:
    """One image model as the list and the picker show it: its download per
    DiT quant on this Mac, the class default, and the Mac it needs when
    this one makes no image with it."""
    name: str
    title: str
    use: str
    blurb: str
    speed: str
    license: str                     # as the user reads it
    license_detail: str              # what the license limits, or ""
    size: str                        # the image size a default pull makes here
    default_quant: str
    quants: tuple                    # (DiT quant, download bytes), ascending
    needs_class: int | None = None   # None when this Mac makes an image

    @property
    def download(self) -> int:
        return dict(self.quants)[self.default_quant]


class NoResidentRung(Exception):
    """The entry has no quant that runs on this Mac."""


@dataclass(frozen=True)
class Seed:
    """The hand-written part of an entry. ``quants`` lists a no-tier
    entry's quants, or the compact candidates of a tiered entry when not
    every sub-4-bit file should show (empty = all of them)."""
    name: str
    title: str
    repo: str
    preferred: str
    blurb: str
    speed: str
    aliases: tuple = ()
    mmproj: str | None = None
    drafter_repo: str = ""
    drafter_file: str = ""
    cross_base: bool = False
    ignored_files: tuple = ()
    license_note: str = ""
    license_detail: str = ""
    quants: tuple = ()
    params: str = ""          # over the header's size label when that is not plain
    use: str = ""             # the "good for" column: 31 characters at most, its width at 80 columns
    preferred_at: tuple = ()  # (memory class, tier) pairs: where a bare pull starts there


_SEED = (
    Seed("qwen3.5-4b", "Qwen3.5-4B", "unsloth/Qwen3.5-4B-GGUF", "high",
         "A small all-rounder for chat and questions about images.",
         "Quick on any Mac: the smallest model here.",
         mmproj="mmproj-F16.gguf",
         use="chat and questions about images"),
    Seed("ternary-bonsai-2-9b", "Ternary Bonsai 2 9B", "prism-ml/Ternary-Bonsai-2-9B-gguf",
         "PQ2_0-MTP-Q8_0",
         "A 9B chat model squeezed to under 2 bits, images included.",
         "Very quick: it reads under 3 GB of weights per token.",
         aliases=("bonsai-2-9b",),
         mmproj="Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf",
         ignored_files=("Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf",),
         quants=("PTQ1_0", "PQ2_0", "Q2_0", "PQ2_0-MTP-Q8_0"),
         use="chat on a small Mac"),
    Seed("nemotron-3.5-bolt-e4b", "Nemotron 3.5 Bolt E4B",
         "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", "high",
         "Built for tool calls and voice assistants.",
         "Quick: a 4B-class model with faster replies on by default.",
         aliases=("bolt-e4b",),
         mmproj="mmproj-Q8_0.gguf",
         drafter_repo="unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF",
         drafter_file="dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf",
         license_note="NVIDIA Open Model License",
         license_detail="Its faster-replies file is under NVIDIA's evaluation license.",
         quants=("UD-IQ2_M", "UD-IQ3_XXS", "UD-Q3_K_XL"),
         params="E4B",
         use="tool calls and voice assistants"),
    Seed("ornith-1.5-9b", "Ornith-1.5-9B", "ornith-ai/Ornith-1.5-9B-GGUF", "high",
         "A coding model that also reads images.",
         "Brisk for a 9B: its own head gives faster replies.",
         mmproj="mmproj-Ornith-1.5-9B-BF16.gguf",
         use="coding"),
    Seed("qwen3.5-9b", "Qwen3.5-9B", "unsloth/Qwen3.5-9B-MTP-GGUF", "high",
         "Good at chat and questions about images, with a long context.",
         "Brisk for a 9B: its own head gives faster replies.",
         mmproj="mmproj-F16.gguf",
         use="chat with a long context"),
    Seed("gemma-4-12b", "gemma-4-12B", "unsloth/gemma-4-12b-it-GGUF", "high",
         "Strong general chat with image input.",
         "Steady for a 12B, with faster replies on by default.",
         mmproj="mmproj-F16.gguf",
         drafter_repo="unsloth/gemma-4-12b-it-GGUF",
         drafter_file="mtp-gemma-4-12b-it.gguf",
         use="general chat"),
    Seed("k2-horizon-7b", "K2-Horizon-7B", "IFM/K2-Horizon-7B-GGUF", "high",
         "A chat model that keeps 512K tokens in view.",
         "Quick: a 7B dense model.",
         params="7B",
         use="chat with a 512K context"),
    Seed("gemma-4-26b-a4b", "gemma-4-26B-A4B", "unsloth/gemma-4-26B-A4B-it-GGUF", "balanced",
         "Strong general chat with image input.",
         "Quick for its size: it reads 4B of its 26B weights per token.",
         mmproj="mmproj-F16.gguf",
         drafter_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
         drafter_file="mtp-gemma-4-26B-A4B-it.gguf",
         use="general chat"),
    Seed("qwen3.8-27b", "Qwen3.8-27B", "unsloth/Qwen3.8-27B-GGUF", "high",
         "Strong at long documents, code and tool calls.",
         "Steady for a 27B, with faster replies on by default.",
         mmproj="mmproj-F16.gguf",
         drafter_repo="incoai/Qwen3.8-27B-DFlash2-GGUF",
         drafter_file="Qwen3.8-27B-DFlash2-Q8_0.gguf",
         use="documents, code and tool calls"),
    Seed("ternary-bonsai-2-27b", "Ternary Bonsai 2 27B", "prism-ml/Ternary-Bonsai-2-27B-gguf",
         "PQ2_0",
         "A 27B chat model squeezed to under 2 bits, images included.",
         "Steady: a 27B, but under 7 GB of weights per token.",
         aliases=("bonsai-2-27b",),
         mmproj="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
         drafter_repo="incoai/Qwen3.8-27B-DFlash2-GGUF",
         drafter_file="Qwen3.8-27B-DFlash2-Q8_0.gguf",
         cross_base=True,
         quants=("PTQ1_0", "PQ2_0"),
         use="27B-class chat on a small Mac"),
    Seed("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF", "balanced",
         "Good general chat with image input.",
         "Quick for its size: it reads 3B of its 35B weights per token.",
         mmproj="mmproj-F16.gguf",
         use="general chat"),
    # bartowski's repo, not the publisher's: it ships the IQ4_XS and sub-4-bit
    # files that put the model in memory on a 32 GB Mac (asher, 2026-10-25).
    Seed("ornith-1.5-35b-a3b", "Ornith-1.5-35B-A3B", "bartowski/Ornith-1.5-35B-A3B-GGUF",
         "balanced",
         "A coding model that also reads images.",
         "Quick: it reads 3B of its 35B weights per token, with faster replies.",
         mmproj="mmproj-Ornith-1.5-35B-A3B-f16.gguf",
         quants=("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "Q2_K", "IQ3_XXS", "Q3_K_S",
                 "IQ3_XS", "Q3_K_M", "IQ3_M", "Q3_K_L"),
         params="35B-A3B",
         use="coding"),
    Seed("k2-horizon-mova-36b-a4b", "K2-Horizon-MoVA-36B", "IFM/K2-Horizon-MoVA-36B-A4B-GGUF",
         "balanced",
         "A chat model that keeps 512K tokens in view.",
         "Quick for its size: it reads 4B of its 36B weights per token.",
         aliases=("k2-horizon-mova",),
         params="36B-A4B",
         use="chat with a 512K context"),
    Seed("ornith-1.5-122b", "Ornith-1.5-122B", "unsloth/Ornith-1.5-122B-GGUF", "balanced",
         "A strong coding model that also reads images.",
         "Quick for its size: it reads 10B of its 122B weights per token.",
         mmproj="mmproj-F16.gguf",
         params="122B-A10B",
         use="coding"),
    Seed("nemotron-3.5-super-vl", "Nemotron 3.5 Super VL",
         "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF", "balanced",
         "Reads long documents and images well.",
         "Quick for its size: it reads 12B of its 120B weights per token.",
         aliases=("nemotron-3.5-super-vl-120b-a12b",),
         mmproj="mmproj-BF16.gguf",
         ignored_files=("LiLiCorr/lilicorr-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf",),
         license_note="NVIDIA Open Model License",
         params="120B-A12B",
         use="long documents and images"),
    Seed("qwen3.8-flash-next", "Qwen3.8-Flash-Next", "unsloth/Qwen3.8-Flash-Next-GGUF",
         "balanced",
         "Strong at long documents, code and tool calls.",
         "Quick for its size: it reads a small part of its weights per token.",
         aliases=("flash-next",),
         mmproj="mmproj-F16.gguf",
         drafter_repo="unsloth/Qwen3.8-Flash-Next-GGUF",
         drafter_file="MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
         license_note="Qwen Community License",
         # At 128 GB the balanced file wires 80 GB and reads its 27 GB table
         # through the page cache, 107 GB in all; the fastest tier holds the
         # whole model in 91 GB and leaves room for other apps.
         preferred_at=((128, "fastest"),),
         use="documents, code and tool calls"),
    Seed("glm-5.3-flash", "GLM-5.3-Flash", "unsloth/GLM-5.3-Flash-GGUF", "balanced",
         "A strong coding and agent model.",
         "Quick for its size: it reads a small part of its weights per token.",
         mmproj="mmproj-F16.gguf",
         use="coding and agents"),
    Seed("deepseek-v4-flash-0731", "DeepSeek-V4-Flash", "unsloth/DeepSeek-V4-Flash-0731-GGUF",
         "balanced",
         "Strong general chat and code.",
         "Quick for its size: it reads a small part of its weights per token.",
         aliases=("deepseek-v4-flash",),
         ignored_files=("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                        "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf"),
         use="chat and code"),
    Seed("deepseek-v4.1-flash-1017", "DeepSeek-V4.1-Flash", "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
         "balanced",
         "Strong at code and agent work, with a 1M token context.",
         "Quick for its size: it reads a small part of its weights per token.",
         aliases=("deepseek-v4.1-flash",),
         mmproj="mmproj-F16.gguf",
         ignored_files=("dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf",
                        "dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf"),
         use="code and agents, 1M context"),
    Seed("glm-5.3", "GLM-5.3", "unsloth/GLM-5.3-GGUF", "balanced",
         "The strongest coding and agent model here.",
         "Quick for its size: it reads a small part of its weights per token.",
         license_note="GLM-5.3 License",
         use="top coding and agents"),
)

# Memory class -> the entries to lead with, chat pick first, then coding,
# then vision. A Mac uses the largest class at or under its RAM, and the
# class below when no name on its list runs in memory here.
RECOMMENDED = {
    8:   ("ternary-bonsai-2-9b", "nemotron-3.5-bolt-e4b", "qwen3.5-4b"),
    16:  ("gemma-4-12b", "nemotron-3.5-bolt-e4b", "k2-horizon-7b",
          "ornith-1.5-9b", "qwen3.5-9b"),
    24:  ("gemma-4-26b-a4b", "qwen3.8-27b", "ternary-bonsai-2-27b"),
    32:  ("qwen3.8-27b", "ornith-1.5-35b-a3b"),
    48:  ("qwen3.8-27b", "ornith-1.5-35b-a3b", "k2-horizon-mova-36b-a4b"),
    64:  ("qwen3.8-27b", "ornith-1.5-35b-a3b", "k2-horizon-mova-36b-a4b"),
    96:  ("qwen3.8-27b", "ornith-1.5-122b", "qwen3.8-flash-next"),
    # From 128 GB the lists end with two picks for speed: the dense 27B and
    # a MoE under 40B with a native head, both far under the bar there.
    128: ("qwen3.8-flash-next", "ornith-1.5-122b", "qwen3.8-27b", "ornith-1.5-35b-a3b"),
    192: ("qwen3.8-flash-next", "glm-5.3-flash", "qwen3.8-27b", "ornith-1.5-35b-a3b"),
    256: ("qwen3.8-flash-next", "glm-5.3-flash", "deepseek-v4.1-flash-1017",
          "qwen3.8-27b", "ornith-1.5-35b-a3b"),
    512: ("deepseek-v4.1-flash-1017", "glm-5.3", "qwen3.8-27b", "ornith-1.5-35b-a3b"),
}


def _params_label(seed: Seed, facts: dict) -> str:
    """The parameter count as the user reads it: the seed's word, else the
    header's size label when it is plain (``27B``, ``122B-A10B``), else
    the counted total (a MoE label such as ``288x10B`` says nothing to a
    user)."""
    if seed.params:
        return seed.params
    label = facts.get("params", "")
    if re.fullmatch(r"\d+(\.\d+)?[BM](-A\d+(\.\d+)?[BM])?", label):
        return label
    n = facts.get("n_params", 0)
    return f"{n / 1e9:.0f}B" if n else label


def _build(seed: Seed, facts: dict) -> Model:
    rungs = tuple(sorted((Rung(quant=q, tier=t, path=p, bytes=b, text_bytes=tb,
                               table_bytes=lb, streams_from_gib=s, head=h)
                          for q, t, p, b, tb, lb, s, h in facts.get("rungs", ())),
                         key=lambda r: r.bytes))
    drafter = None
    if seed.drafter_file:
        kind, revision, nbytes = facts.get("drafter") or ("", "", 0)
        drafter = Drafter(kind=kind, repo=seed.drafter_repo, revision=revision,
                          file=seed.drafter_file, bytes=nbytes, cross_base=seed.cross_base)
    kv = facts.get("kv_cost")
    return Model(
        name=seed.name, title=seed.title, repo=seed.repo,
        revision=facts.get("revision", "main"), rungs=rungs, preferred=seed.preferred,
        params=_params_label(seed, facts), blurb=seed.blurb, speed=seed.speed, use=seed.use,
        license=facts.get("license", ""), license_url=facts.get("license_url", ""),
        license_note=seed.license_note or facts.get("license_name", ""),
        license_detail=seed.license_detail, aliases=seed.aliases, arch=facts.get("arch", ""), moe=facts.get("moe", False),
        native_mtp=facts.get("native_mtp", False),
        mmproj=seed.mmproj, mmproj_bytes=facts.get("mmproj_bytes", 0) if seed.mmproj else 0,
        drafter=drafter, ignored_files=seed.ignored_files,
        kv_cost=KvCost(tuple(tuple(row) for row in kv)) if kv else None,
        heads=facts.get("heads", 0), trained_ctx=facts.get("trained_ctx", 0),
        preferred_at=seed.preferred_at)


def _models() -> dict:
    from gmlx.load.catalog_facts import FACTS

    return {s.name: _build(s, FACTS.get(s.name, {})) for s in _SEED}


MODELS: dict = _models()
SEEDS: dict = {s.name: s for s in _SEED}


def model(name: str) -> Model | None:
    """The entry called ``name``, by name or alias."""
    key = (name or "").strip().lower()
    m = MODELS.get(key)
    if m is not None:
        return m
    for m in MODELS.values():
        if key in m.aliases:
            return m
    return None


def closest_name(text: str) -> str | None:
    """The entry name closest to a typo, or None."""
    names = {}
    for m in MODELS.values():
        names[m.name] = m.name
        for a in m.aliases:
            names[a] = m.name
    hit = difflib.get_close_matches((text or "").lower(), list(names), n=1, cutoff=0.6)
    return names[hit[0]] if hit else None


def license_text(m: Model) -> str:
    """The license as the user reads it."""
    if m.license_note and m.license not in LICENSE_WORDS:
        return m.license_note
    return LICENSE_WORDS.get(m.license, m.license or "license unknown")


# -- this Mac -------------------------------------------------------------

def memory_class(ram_bytes: int) -> int:
    """The memory class of a Mac with ``ram_bytes`` of RAM."""
    gib = ram_bytes / GiB
    fit = [c for c in MEMORY_CLASSES if c <= gib + 0.5]
    return fit[-1] if fit else MEMORY_CLASSES[0]


def box_for_class(cls: int) -> Box:
    """The Box of a stock Mac in memory class ``cls``, without Metal: the
    planner's working set, 0.9 of it as the resident budget."""
    from gmlx.gen.prefill_decay import cap_bytes_for
    from gmlx.serve.capacity import working_set_for_ram
    from gmlx.stream.plan import ceiling_for

    ram = cls * GiB
    ws = working_set_for_ram(ram)
    return Box(ram=ram, ws=int(ws), budget=int(0.9 * ws),
               ceiling=int(ceiling_for(ram, ws)), cap=int(cap_bytes_for(ws)))


def box_for_this_mac() -> Box:
    """This Mac, from Metal when it answers, else from its RAM alone.
    ``GMLX_PULL_RAM_GB=N`` answers for a stock N GB Mac instead, so the
    picker can be read at another memory size."""
    pretend = os.environ.get("GMLX_PULL_RAM_GB", "")
    if pretend:
        if not pretend.isdecimal() or int(pretend) == 0:
            raise ValueError(f"GMLX_PULL_RAM_GB={pretend!r} is not a whole number of GB")
        return box_for_class(int(pretend))
    from gmlx.gen.prefill_decay import cap_bytes_for
    from gmlx.load.memfit import total_ram_bytes
    from gmlx.serve.capacity import working_set_bytes, working_set_for_ram
    from gmlx.stream.plan import ceiling_for

    ram = total_ram_bytes() or (16 * GiB)
    ws = working_set_bytes()
    if ws:
        from gmlx.device import stream_budget_for
        budget = stream_budget_for(ws)
    else:
        ws = working_set_for_ram(ram)
        budget = int(0.8 * ram)
    return Box(ram=int(ram), ws=int(ws), budget=int(budget),
               ceiling=int(ceiling_for(ram, ws)), cap=int(cap_bytes_for(ws)))


# -- rungs ---------------------------------------------------------------

def has_tiers(m: Model) -> bool:
    return any(r.tier for r in m.rungs)


def tiers(m: Model) -> list:
    """The tier rungs in TIERS order."""
    by_tier = {r.tier: r for r in m.rungs if r.tier}
    return [by_tier[t] for t in TIERS if t in by_tier]


def compact_candidates(m: Model) -> list:
    """The compact quants of a tiered entry, ascending."""
    if not has_tiers(m):
        return []
    return [r for r in m.rungs if not r.tier]


def preferred_tier(m: Model, box: Box | None = None) -> str:
    """The tier, or quant, a bare pull starts from: the seed's word for
    the memory class of ``box`` when it names one, else ``preferred``."""
    if box is not None:
        cls = memory_class(box.ram)
        for c, want in m.preferred_at:
            if c == cls:
                return want
    return m.preferred


def preferred_rung(m: Model, box: Box | None = None) -> Rung | None:
    want = preferred_tier(m, box)
    for r in m.rungs:
        if r.tier == want or (not r.tier and r.quant == want):
            return r
    return None


def has_head(m: Model, r: Rung) -> bool:
    return m.native_mtp or r.head


def speculates(m: Model, r: Rung, drafter_on: bool, streaming: bool = False) -> bool:
    """Whether a plan on ``r`` speculates: never from the SSD; else with a
    native head, or with the drafter add-on on."""
    if streaming:
        return False
    return has_head(m, r) or (drafter_on and m.drafter is not None)


def resident_bytes(m: Model, r: Rung, speculative: bool) -> int:
    """The weights a plan holds in memory: the text tensors less the
    tables a load reads from the file, plus a native head's block when
    the plan speculates."""
    n = (r.text_bytes or r.bytes) - r.table_bytes
    if speculative and has_head(m, r) and r.text_bytes:
        n += r.bytes - r.text_bytes
    return n


def addon_bytes(m: Model, mmproj: bool, drafter: bool) -> int:
    n = 0
    if mmproj and m.mmproj:
        n += m.mmproj_bytes
    if drafter and m.drafter is not None:
        n += m.drafter.bytes
    return n


def rung_state(m: Model, r: Rung, box: Box) -> str:
    """RESIDENT, STREAMS or OVER, from the file alone."""
    from gmlx.stream.plan import VERDICT_RESIDENT, VERDICT_TABLES, weight_verdict

    v = weight_verdict(r.text_bytes or r.bytes, r.table_bytes, box.budget, box.ram)
    if v in (VERDICT_RESIDENT, VERDICT_TABLES):
        return RESIDENT
    if r.streams_from_gib is not None and r.streams_from_gib * GiB <= box.ram:
        return STREAMS
    return OVER


def usable(state: str) -> bool:
    return state in (RESIDENT, STREAMS)


def fits_with(m: Model, r: Rung, box: Box, addon: int, speculative: bool) -> bool:
    """The default plan's test: the weights and the add-ons under the bar."""
    return resident_bytes(m, r, speculative) + addon <= box.budget


def needs_class(m: Model, r: Rung) -> int | None:
    """The smallest memory class where ``r`` runs, or None."""
    for cls in MEMORY_CLASSES:
        if rung_state(m, r, box_for_class(cls)) != OVER:
            return cls
    return None


def needs_class_model(m: Model) -> int | None:
    """The smallest memory class where some rung of ``m`` runs."""
    for cls in MEMORY_CLASSES:
        box = box_for_class(cls)
        if any(rung_state(m, r, box) != OVER for r in m.rungs):
            return cls
    return None


# -- the context figure ---------------------------------------------------

def kv_costs(kc: KvCost) -> list:
    """``mem_preflight`` cost entries from a recorded KvCost."""
    from gmlx.serve.mem_preflight import FixedRows, StepTokens

    out = []
    for kind, n, bpt, count in kc.costs:
        window = {"full": None, "window": int(n), "rows": FixedRows(n),
                  "step": StepTokens(n)}[kind]
        out.extend([(window, bpt)] * int(count))
    return out


def serialise_costs(costs) -> KvCost:
    """A KvCost from ``boot_costs``' cost entries, identical rows counted."""
    from gmlx.serve.mem_preflight import FixedRows, StepTokens

    rows: dict = {}
    for window, bpt in costs:
        if window is None:
            key = ("full", 0, float(bpt))
        elif isinstance(window, FixedRows):
            key = ("rows", int(window), float(bpt))
        elif isinstance(window, StepTokens):
            key = ("step", int(window), float(bpt))
        else:
            key = ("window", int(window), float(bpt))
        rows[key] = rows.get(key, 0) + 1
    return KvCost(tuple((k[0], k[1], k[2], n) for k, n in rows.items()))


def context_estimate(m: Model, r: Rung, box: Box, addon: int, speculative: bool) -> int | None:
    """The largest context, a multiple of 1024 tokens and at most the
    trained context, at which the weights, the add-ons, the cache and a
    prefill allowance fit under the ceiling. 0 under 1K; None when the
    entry has no recorded cost."""
    if m.kv_cost is None or m.heads <= 0 or m.trained_ctx <= 0:
        return None
    from gmlx.serve.mem_preflight import prompt_kv_bytes

    costs = kv_costs(m.kv_cost)
    fixed = resident_bytes(m, r, speculative) + addon

    def fits(t: int) -> bool:
        s = min(2048, t)
        return (fixed + prompt_kv_bytes(costs, t)
                + min(box.cap, m.heads * s * (t + s) * 2) <= box.ceiling)

    top = m.trained_ctx // K
    if top < 1 or not fits(K):
        return 0
    if fits(top * K):
        return top * K
    lo, hi = 1, top
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid * K):
            lo = mid
        else:
            hi = mid
    return lo * K


def context_text(m: Model, r: Rung, box: Box, addon: int, speculative: bool) -> str:
    est = context_estimate(m, r, box, addon, speculative)
    if est is None:
        return ""
    if est < K:
        return "Almost no room for context."
    return f"Room for about {est // K}K tokens of context."


def _room_words(est: int | None) -> str:
    if est is None:
        return "in memory"
    if est < K:
        return "almost no room"
    return f"room for ~{est // K}K"


# -- the default plan and what shows --------------------------------------

def _addon_sets(m: Model) -> list:
    """The add-on sets the default walk tries: all on, drafter off, both off."""
    out = []
    for mm, dr in ((True, True), (True, False), (False, False)):
        pair = (mm and m.mmproj is not None, dr and m.drafter is not None)
        if pair not in out:
            out.append(pair)
    return out


def _ladder(m: Model, box: Box | None = None) -> list:
    """The preferred rung on ``box``, then each lower one, largest first."""
    pref = preferred_rung(m, box)
    if pref is None:
        return []
    pool = tiers(m) if has_tiers(m) else list(m.rungs)
    return [pref] + sorted((r for r in pool if r.bytes < pref.bytes),
                           key=lambda r: r.bytes, reverse=True)


def default_plan(m: Model, box: Box) -> PullPlan | None:
    """The rung and add-on toggles a bare pull takes on this Mac. Quality
    drops before speed and speed before images: every add-on on, then the
    drafter off, then both off, and within each the preferred tier, the
    lower tiers and the largest compact quant that fits. When nothing
    fits in memory, the lowest tier that streams, else the largest
    compact quant that streams, with both add-ons off."""
    compacts = sorted(compact_candidates(m), key=lambda r: r.bytes, reverse=True)
    for mm, dr in _addon_sets(m):
        addon = addon_bytes(m, mm, dr)
        for r in _ladder(m, box) + compacts:
            if (rung_state(m, r, box) == RESIDENT
                    and fits_with(m, r, box, addon, speculates(m, r, dr))):
                return PullPlan(m, r, mm, dr, False)
    ladder = tiers(m) if has_tiers(m) else list(m.rungs)
    for r in sorted(ladder, key=lambda r: r.bytes) + compacts:
        if rung_state(m, r, box) == STREAMS:
            return PullPlan(m, r, False, False, True)
    return None


def compact_rung(m: Model, box: Box) -> Rung | None:
    """The one compact quant shown here: the default plan's when that is
    one; else the largest in memory, only when no tier is; else the
    largest that streams, only when no tier runs at all."""
    compacts = compact_candidates(m)
    if not compacts:
        return None
    dp = default_plan(m, box)
    if dp is not None and not dp.rung.tier:
        return dp.rung
    states = [rung_state(m, t, box) for t in tiers(m)]
    if RESIDENT not in states:
        for r in reversed(compacts):
            if rung_state(m, r, box) == RESIDENT:
                return r
    if not any(usable(s) for s in states):
        for r in reversed(compacts):
            if rung_state(m, r, box) == STREAMS:
                return r
    return None


def shown_rungs(m: Model, box: Box) -> tuple:
    """The rungs the configure screen lists: the tiers that run here and
    the compact quant, or a no-tier entry's quants that run here."""
    if not has_tiers(m):
        return tuple(r for r in m.rungs if usable(rung_state(m, r, box)))
    out = [r for r in tiers(m) if usable(rung_state(m, r, box))]
    c = compact_rung(m, box)
    if c is not None:
        out.append(c)
    return tuple(sorted(out, key=lambda r: r.bytes))


def rung_label(m: Model, r: Rung, box: Box, addon: int, speculative: bool) -> tuple:
    """``(label, note)`` for a rung as the user reads it."""
    state = rung_state(m, r, box)
    if state == STREAMS:
        return "from SSD, slow", ""
    if state == OVER:
        cls = needs_class(m, r)
        return (f"needs a {cls} GB Mac" if cls else "too big for a Mac", "")
    room = _room_words(context_estimate(m, r, box, addon, speculative))
    if r == compact_rung(m, box):
        return f"compact, {room}", "smaller, lower quality"
    return room, ""


def tags(m: Model) -> list:
    out = []
    if m.mmproj:
        out.append("images")
    if m.drafter is not None or m.native_mtp or any(r.head for r in m.rungs):
        out.append("faster replies")
    return out


# -- groups ----------------------------------------------------------------

def recommended_for(box: Box) -> list:
    """The entries to lead with on this Mac: the names of the largest
    class at or under its RAM whose default plan runs in memory, in
    order, walking down a class when none does."""
    cls = memory_class(box.ram)
    for c in reversed([c for c in MEMORY_CLASSES if c <= cls]):
        picks = []
        for name in RECOMMENDED.get(c, ()):
            m = MODELS.get(name)
            if m is None:
                continue
            dp = default_plan(m, box)
            if dp is not None and not dp.streaming:
                picks.append(m)
        if picks:
            return picks
    return []


def groups_for(box: Box) -> tuple:
    """``(recommended, fits, streams, too_big)``, each a list of entries."""
    rec = recommended_for(box)
    fits, streams, too_big = [], [], []
    for m in MODELS.values():
        if m in rec:
            continue
        states = [rung_state(m, r, box) for r in m.rungs]
        if RESIDENT in states:
            fits.append(m)
        elif STREAMS in states:
            streams.append(m)
        else:
            too_big.append(m)

    def download(m):
        dp = default_plan(m, box)
        return -plan_bytes(dp) if dp else 0
    fits.sort(key=download)
    streams.sort(key=download)
    too_big.sort(key=lambda m: (needs_class_model(m) or 0, m.name))
    return rec, fits, streams, too_big


# -- plans -----------------------------------------------------------------

IMAGE_GROUP_TITLE = "Image models"
_IMAGE_LICENSES = {"apache-2.0": "Apache 2.0", "qwen-research": "Qwen Research License",
                   "krea-2-community": "Krea 2 Community License",
                   "flux-non-commercial-license": "FLUX Non-Commercial License",
                   "flux-kommunity": "FLUX Kommunity License"}


def image_entries(box: Box) -> tuple:
    """``(runs, too_big)``: the image models that make an image on this
    Mac, in catalog order, and those that need a larger one. A Mac under
    the smallest image memory class runs none."""
    from gmlx.image import catalog as ic

    ram = box.ram
    cls = ic.memory_class(ram)
    small = ram / GiB + 0.5 < ic.MEMORY_CLASSES[0]
    runs, too_big = [], []
    for r in ic.RECIPES.values():
        ok = not small and ic.makes_image(r, cls)
        quants = tuple(sorted(((q, ic.download_bytes(r, ram, q)) for q in r.dit_quants),
                              key=lambda qb: qb[1]))
        entry = ImageEntry(
            name=r.name, title=r.title, use=r.use, blurb=r.blurb, speed=r.speed,
            license=_IMAGE_LICENSES.get(r.license, r.license), license_detail=r.license_note,
            size=ic.image_size(r, ram), default_quant=ic.default_quants(r, ram)[0],
            quants=quants,
            needs_class=None if ok else (ic.MEMORY_CLASSES[0] if small else ic.least_class(r)))
        (runs if ok else too_big).append(entry)
    return runs, too_big


def image_command(e: ImageEntry, quant: str | None = None) -> str:
    """``gmlx pull <name>``, with ``--quant`` off the default."""
    cmd = f"gmlx pull {e.name}"
    if quant and quant != e.default_quant:
        cmd += f" --quant {quant}"
    return cmd


def image_rows(e: ImageEntry) -> list:
    """One line per DiT quant: the quant, its download here and the default mark."""
    return [(f"{q:<9} {'':<16} {human_gb(n):>9}  "
             + ("  default" if q == e.default_quant else "")).rstrip()
            for q, n in e.quants]


def image_entry_text(e: ImageEntry) -> str:
    """One image model for ``gmlx pull <name> --list``: its DiT quants with
    their downloads here, and the command."""
    lines = [f"{e.title}  {e.size}  {e.license}".rstrip(), e.blurb, e.speed]
    if e.license_detail:
        detail = e.license_detail[0].upper() + e.license_detail[1:]
        lines.append(detail if detail.endswith(".") else detail + ".")
    lines += ["", "Quality"] + [f"  {row}" for row in image_rows(e)]
    if e.needs_class is None:
        lines.append(image_command(e))
    else:
        lines.append(f"Needs a {e.needs_class} GB Mac.")
    return "\n".join(lines)


def _norm_quant(q: str) -> str:
    q = (q or "").strip().upper()
    return q[3:] if q.startswith("UD-") else q


def find_rung(m: Model, quant: str) -> Rung | None:
    """The rung a ``--quant`` value names: a tier, or a quant with or
    without its ``UD-`` prefix."""
    want = (quant or "").strip().lower()
    for r in m.rungs:
        if r.tier and r.tier == want:
            return r
    for r in m.rungs:
        if _norm_quant(r.quant) == _norm_quant(quant):
            return r
    return None


def quant_choices(m: Model) -> str:
    """The tiers and quants ``--quant`` accepts, for an error."""
    names = [r.tier for r in tiers(m)] + [r.quant for r in m.rungs if not r.tier or not has_tiers(m)]
    return ", ".join(dict.fromkeys(names))


def plan_for(m: Model, box: Box, *, quant: str | None = None,
             mmproj: bool | None = None, drafter: bool | None = None) -> PullPlan:
    """The plan a named pull makes. ``quant`` None takes the default
    plan's rung; ``mmproj`` and ``drafter`` None take its toggles (both
    off from the SSD). Raises NoResidentRung when the entry has no default
    here and no quant was named, ValueError for a bad quant or a toggle
    the entry has no file for."""
    dp = default_plan(m, box)
    if quant is None:
        if dp is None:
            smallest = min(m.rungs, key=lambda r: r.bytes) if m.rungs else None
            hint = (f"; --quant {smallest.quant} pulls its smallest file "
                    f"({human_gb(smallest.bytes)}) anyway" if smallest else "")
            raise NoResidentRung(f"{m.name} has no quant that runs on this Mac{hint}")
        rung = dp.rung
    else:
        rung = find_rung(m, quant)
        if rung is None:
            raise ValueError(f"{m.name} has no quality tier or quant {quant!r}; "
                             f"choose from {quant_choices(m)}")
    if mmproj and m.mmproj is None:
        raise ValueError(f"{m.name} has no vision tower")
    if drafter is not None and m.drafter is None:
        flag = "--drafter" if drafter else "--no-drafter"
        raise ValueError(f"{m.name} comes without faster replies; drop {flag}")
    state = rung_state(m, rung, box)
    streaming = state != RESIDENT and m.moe
    if streaming:
        return PullPlan(m, rung, bool(mmproj), False, True)
    if dp is not None and not dp.streaming:
        d_mm, d_dr = dp.mmproj, dp.drafter
    else:
        d_mm, d_dr = False, False
    return PullPlan(m, rung,
                    d_mm if mmproj is None else bool(mmproj),
                    d_dr if drafter is None else bool(drafter), False)


def plan_command(plan: PullPlan, box: Box) -> str:
    """The short ``gmlx pull`` command that reproduces ``plan``."""
    m = plan.model
    dp = default_plan(m, box)
    parts = ["gmlx pull", m.name]
    if dp is None or plan.rung != dp.rung:
        parts += ["--quant", plan.rung.tier or plan.rung.quant]
    if plan.streaming:
        if plan.mmproj:
            parts.append("--vision")
        return " ".join(parts)
    d_mm, d_dr = (dp.mmproj, dp.drafter) if dp is not None and not dp.streaming else (False, False)
    if m.mmproj and plan.mmproj != d_mm:
        parts.append("--vision" if plan.mmproj else "--no-vision")
    if m.drafter is not None and plan.drafter != d_dr:
        parts.append("--drafter" if plan.drafter else "--no-drafter")
    return " ".join(parts)


def plan_addon_bytes(plan: PullPlan) -> int:
    return addon_bytes(plan.model, plan.mmproj, plan.drafter)


def plan_speculates(plan: PullPlan) -> bool:
    return speculates(plan.model, plan.rung, plan.drafter, plan.streaming)


def plan_bytes(plan: PullPlan) -> int:
    """The whole download."""
    return plan.rung.bytes + plan_addon_bytes(plan)


def rung_ref(m: Model, r: Rung) -> str:
    return f"hf:{m.repo}/{r.path}@{m.revision}"


def plan_refs(plan: PullPlan) -> list:
    """The ``hf:`` refs a plan downloads: the model, then its add-ons."""
    m = plan.model
    refs = [rung_ref(m, plan.rung)]
    if plan.mmproj and m.mmproj:
        refs.append(f"hf:{m.repo}/{m.mmproj}@{m.revision}")
    if plan.drafter and m.drafter is not None:
        d = m.drafter
        refs.append(f"hf:{d.repo}/{d.file}@{d.revision}")
    return refs


def dest_path(repo: str, path_in_repo: str, base_dir: str, nest: bool) -> str:
    """Where a pull puts a repo file: under ``<org>__<repo>/`` when nested,
    keeping the file's folder inside the repo, as the download does."""
    sub = repo.replace("/", "__") if nest else ""
    return os.path.join(base_dir, sub, path_in_repo)


def plan_pairs(plan: PullPlan, base_dir: str, nest: bool) -> dict:
    """What registration pairs with the model file: ``{model path:
    {"mmproj": path, "draft_gguf": path, "alias": name}}``."""
    m = plan.model
    from gmlx.load.preflight import shard_names

    first = shard_names(os.path.basename(plan.rung.path))[0]
    model_path = dest_path(m.repo, first, base_dir, nest)
    pairs: dict = {"alias": m.name}
    if plan.mmproj and m.mmproj:
        pairs["mmproj"] = dest_path(m.repo, m.mmproj, base_dir, nest)
    if plan.drafter and m.drafter is not None:
        pairs["draft_gguf"] = dest_path(m.drafter.repo, m.drafter.file, base_dir, nest)
    return {model_path: pairs}


def local_state(plan: PullPlan, base_dir: str, nest: bool) -> tuple:
    """``(on_disk_bytes, remaining_bytes, per_ref)`` from recorded sizes
    and the files at the destination, with no network. ``per_ref`` maps
    each ref to ``(on_disk, total, complete)``."""
    from gmlx.load.preflight import shard_names

    m = plan.model
    items = [(rung_ref(m, plan.rung), m.repo, plan.rung.path, plan.rung.bytes)]
    if plan.mmproj and m.mmproj:
        items.append((f"hf:{m.repo}/{m.mmproj}@{m.revision}", m.repo, m.mmproj, m.mmproj_bytes))
    if plan.drafter and m.drafter is not None:
        d = m.drafter
        items.append((f"hf:{d.repo}/{d.file}@{d.revision}", d.repo, d.file, d.bytes))
    per_ref, on_disk, remaining = {}, 0, 0
    for ref, repo, path, total in items:
        have, complete = 0, True
        for name in shard_names(path):
            p = dest_path(repo, name, base_dir, nest)
            if os.path.isfile(p):
                have += os.path.getsize(p)
            else:
                complete = False
                if os.path.isfile(p + ".part"):
                    have += os.path.getsize(p + ".part")
        have = min(have, total) if total else have
        per_ref[ref] = (have, total, complete)
        on_disk += have
        remaining += max(0, total - have)
    return on_disk, remaining, per_ref


# -- text -------------------------------------------------------------------

def human_gb(n: int) -> str:
    from gmlx.serve.lifecycle import human_gb as _human_gb

    return _human_gb(n)


def rung_rows(m: Model, box: Box, dp: PullPlan | None, addon: int) -> list:
    """One line per quality rung: tier, quant, the file's size, its state or
    room for context here, and the default mark."""
    dr = dp.drafter if dp else False
    pool = m.rungs if not has_tiers(m) else tiers(m)
    rows = []
    for r in sorted(set(pool) | set(shown_rungs(m, box)), key=lambda r: r.bytes):
        label, note = rung_label(m, r, box, addon, speculates(m, r, dr))
        mark = "  default" if dp is not None and r == dp.rung else ""
        name = r.tier or "compact" if has_tiers(m) else r.quant
        quant = r.quant if has_tiers(m) else ""
        rows.append(f"{name:<9} {quant:<16} {human_gb(r.bytes):>9}  {label}{mark}"
                    + (f"  ({note})" if note else ""))
    return rows


def extras_text(m: Model) -> str:
    """``image input 0.8 GB, faster replies 2.6 GB``, or "" for an entry with neither."""
    extras = []
    if m.mmproj:
        extras.append(f"image input {human_gb(m.mmproj_bytes)}")
    if m.drafter is not None:
        extras.append(f"faster replies {human_gb(m.drafter.bytes)}")
    return ", ".join(extras)


def entry_text(m: Model, box: Box) -> str:
    """One entry for ``gmlx pull <name> --list``: its quality rungs with
    their state and room for context here, and its add-ons."""
    lines = [f"{m.title}  {m.params}  {license_text(m)}".rstrip(), m.blurb, m.speed]
    if m.license_detail:
        lines.append(m.license_detail)
    lines.append("")
    dp = default_plan(m, box)
    mm = dp.mmproj if dp else False
    dr = dp.drafter if dp else False
    addon = addon_bytes(m, mm, dr)
    lines.append("Quality")
    lines += [f"  {row}" for row in rung_rows(m, box, dp, addon)]
    extras = extras_text(m)
    if extras:
        lines.append("Extras: " + extras)
    if dp is not None:
        lines.append(context_text(m, dp.rung, box, addon, plan_speculates(dp)))
        lines.append(plan_command(dp, box))
    else:
        cls = needs_class_model(m)
        lines.append(f"Needs a {cls} GB Mac." if cls else "Too big for a Mac.")
    return "\n".join(line for line in lines if line is not None)


GROUP_TITLES = ("Recommended for this Mac", "Runs in memory",
                "Runs from the SSD, much slower", "Too big for this Mac")
OTHER_MODELS_LINE = "Other models: gmlx pull <huggingface.co link>, or hf:<org>/<repo>/<file.gguf>"
LIST_FOOTER = ("Each download is the model's default quality tier for this Mac, with image\n"
               "input and faster replies where they fit. gmlx pull <name> --list shows its\n"
               "other tiers, and gmlx pull <name> --quant <tier> picks one.\n"
               + OTHER_MODELS_LINE)


def group_titles(groups: tuple) -> tuple:
    """The four group titles for ``groups``: the second says "Also" when a
    recommended group comes before it."""
    titles = list(GROUP_TITLES)
    if groups[0]:
        titles[1] = "Also runs in memory"
    return tuple(titles)


def images_cell(m: Model) -> str:
    return "yes" if m.mmproj else ""


def _chat_row(m: Model, box: Box, verbose: bool) -> tuple:
    """(name, size cell, images cell, use, detail lines, is a download)."""
    dp = default_plan(m, box)
    if dp is None:
        cls = needs_class_model(m)
        size = f"{cls} GB Mac" if cls else "too big"
    else:
        size = human_gb(plan_bytes(dp))
    detail = []
    if verbose:
        addon = addon_bytes(m, dp.mmproj if dp else False, dp.drafter if dp else False)
        detail = [f"      {row}" for row in rung_rows(m, box, dp, addon)]
        if extras_text(m):
            detail.append(f"      extras: {extras_text(m)}")
    return (m.name, size, images_cell(m), m.use, detail, dp is not None)


def _image_row(e: ImageEntry, verbose: bool) -> tuple:
    """An image model's row: its download here, or the Mac it needs; the
    IMAGES cell stays blank, since every image model edits from a picture."""
    size = human_gb(e.download) if e.needs_class is None else f"{e.needs_class} GB Mac"
    detail = [f"      {row}" for row in image_rows(e)] if verbose else []
    return (e.name, size, "", e.use, detail, e.needs_class is None)


def recommendation_text(groups: tuple, box: Box, *, images=None, verbose: bool = False) -> str:
    """The recommendations as text: one table per group with a header row,
    one line per model: name, download at its default plan, whether it
    reads images, and what it is good for. ``images`` is
    ``image_entries(box)``: the models that make an image here print as an
    "Image models" table after the chat models that run, and the others
    join the too-big table. ``verbose`` adds each model's quality rungs
    and extras under its row, as ``entry_text`` lists them."""
    lines = [f"This Mac has {box.ram // GiB} GB of memory.", ""]
    runs, big = images if images is not None else ((), ())
    titles = group_titles(groups)
    tables = [(titles[i], [_chat_row(m, box, verbose) for m in groups[i]]) for i in range(3)]
    tables.append((IMAGE_GROUP_TITLE, [_image_row(e, verbose) for e in runs]))
    tables.append((titles[3], [_chat_row(m, box, verbose) for m in groups[3]]
                   + [_image_row(e, verbose) for e in big]))
    name_w = max([4, *(len(row[0]) for _t, rows in tables for row in rows)])
    for title, rows in tables:
        if not rows:
            continue
        size_head = "DOWNLOAD" if any(row[5] for row in rows) else "NEEDS"
        size_w = max([len(size_head), *(len(row[1]) for row in rows)])
        lines.append(title)
        lines.append(f"  {'NAME':<{name_w}}  {size_head:<{size_w}}  IMAGES  GOOD FOR")
        for name, size, imgs, use, detail, _download in rows:
            lines.append(f"  {name:<{name_w}}  {size:>{size_w}}  {imgs:<6}  {use}".rstrip())
            lines += detail
        lines.append("")
    lines.append(LIST_FOOTER)
    return "\n".join(lines).rstrip() + "\n"


