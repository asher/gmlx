"""The server e2e config matrix.

Each :class:`Scenario` is one server launch (start mode + config) plus the request
targets fired at it and the post-checks run against the live server. The set
covers the config surface and its high-value combinations:

  core       single-model + config profiles (precedence, @profile, system, negatives)
  kv         quantized KV cache (baseline / 8-bit / 4-bit) under a deep needle recall
  cache      APC prompt cache: disabled / memory-only / SSD-disk / diskxkv8 combo
  residency  multi-model LRU eviction under a tight budget; idle-TTL reaping
  template   chat_template override (config + single-model --chat-template)
  endpoints  /v1/models, /v1/metrics, /unload (one + all), /v1/reload, cache reset
  negative   unknown id -> 404, unknown profile -> 400, HF-gate refuses a stray id
  discovery  --models-dir header-only scan serves derived ids
  vlm        gemma-4-E2B + mmproj image description
  mtp        gemma-4-E2B + assistant drafter speculative; lossless-greedy vs base

The coherence-judged tiers (core/kv/template/endpoints) run on gemma-4-E2B: a 0.6B model
is too weak to follow strict-format prompts, so the judge correctly fails it on baseline
weakness and that noise masks real config regressions. E2B gives a clean baseline where a
config-induced degradation actually stands out. The structural tiers (residency/discovery
/negative) keep the tiny models - they need distinct small sizes for LRU eviction and only
fire the easy `capital` prompt. The cache tier ALSO stays on Qwen3-0.6B: APC block/prefix
reuse needs a plain `KVCache`, and gemma-4's sliding-window `RotatingKVCache` bypasses APC
entirely - so only an APC-capable model can verify the cache feature (its one judged
prompt, anchored `capital`, doesn't need E2B anyway).

A scenario whose required model handles are missing is skipped (not failed).
Pure construction - no model loads here, so the whole matrix builds under --dry-run.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import checks
import prompts as P
from checks import CheckResult


# data model
@dataclass
class ReqTarget:
    name: str
    model: str                       # request "model": id, id@profile, "" (default),
                                     # "__first__" (resolve via /v1/models), or a stray id
    prompts: list = field(default_factory=list)
    sampling: dict = field(default_factory=dict)
    expect_status: int = 200         # 200, or 4xx/5xx for a negative test
    image_handle: Optional[str] = None
    repeat: int = 1                  # determinism / cache-reuse: send each prompt N times
    stream: bool = False


@dataclass
class Scenario:
    key: str
    title: str
    tier: str
    needs: list = field(default_factory=list)        # required model handles
    serve_args: list = field(default_factory=list)   # extra serve args (besides host/port)
    env: dict = field(default_factory=dict)
    config: Optional[dict] = None                    # YAML config dict (runner dumps it)
    targets: list = field(default_factory=list)
    post: list = field(default_factory=list)         # [fn(client) -> list[CheckResult]]
    notes: str = ""


# post-check helpers (closures capture expected values at build time)
def pc_health_ready() -> Callable:
    def _check(client):
        st, body = client.health()
        ok = st == 200 and isinstance(body, dict) and body.get("status") == "healthy"
        return [CheckResult("health", ok, str(body)[:200] if not ok else "healthy")]
    return _check


def pc_models_exactly(expected_ids: set) -> Callable:
    def _check(client):
        st, body = client.models()
        if st != 200 or not isinstance(body, dict):
            return [CheckResult("models_list", False, f"status={st}")]
        ids = {m["id"] for m in body.get("data", [])}
        # configured ids must all appear; nothing from the HF cache may leak in
        missing = expected_ids - ids
        extra = ids - expected_ids
        ok = not missing
        detail = f"ids={sorted(ids)}"
        if missing:
            detail = f"missing={sorted(missing)} got={sorted(ids)}"
        res = [CheckResult("models_list", ok, detail)]
        # extras are allowed only if they are alias entries (alias_of present)
        alias_ids = {m["id"] for m in body.get("data", []) if m.get("alias_of")}
        leak = extra - alias_ids
        res.append(CheckResult("no_hf_cache_leak", not leak,
                               f"unexpected ids {sorted(leak)}" if leak else "clean"))
        return res
    return _check


def pc_apc_enabled(expected: bool) -> Callable:
    def _check(client):
        st, body = client.health()
        got = bool(body.get("apc_enabled")) if isinstance(body, dict) else None
        ok = (st == 200 and got == expected)
        return [CheckResult("apc_enabled", ok, f"expected={expected} got={got}")]
    return _check


def _resident(client) -> list:
    st, body = client.metrics()
    if st != 200 or not isinstance(body, dict):
        return []
    return ((body.get("server") or {}).get("resident_models")) or []


def pc_cache_reuse(model: str, prompt: P.PromptInstance,
                   *, require_exercised: bool = True) -> Callable:
    """Send the same greedy prompt twice; the second must be byte-identical (cache
    must not corrupt output) and a cache counter should advance (cache exercised).

    ``require_exercised=False`` downgrades the counter check to informational: under a
    quantized KV cache, mlx-vlm's APC is bypassed by design (``_cache_entry_supports_
    {block,exact}_apc`` admit only ``KVCache`` & friends, never ``QuantizedKVCache``),
    so no counter moves - the model just recomputes, uncorrupted. We still assert the
    real invariant (no corruption / no crash) and surface the counters for inspection."""
    def _check(client):
        before = client.cache_stats()[1]
        st1, b1 = client.chat(model, prompt.messages, max_tokens=prompt.max_tokens,
                              temperature=0.0)
        st2, b2 = client.chat(model, prompt.messages, max_tokens=prompt.max_tokens,
                              temperature=0.0)
        after = client.cache_stats()[1]
        t1 = checks.extract_chat_text(b1) if isinstance(b1, dict) else None
        t2 = checks.extract_chat_text(b2) if isinstance(b2, dict) else None
        if not (t1 and t2):
            detail = (f"empty/failed response "
                      f"(first={'ok' if t1 else 'empty'}, second={'ok' if t2 else 'empty'})")
        elif t1 == t2:
            detail = "identical"
        else:
            detail = "second response diverged"
        res = [CheckResult("cache_no_corruption", bool(t1) and bool(t2) and t1 == t2,
                           detail)]
        advanced = _stats_advanced(before, after)
        delta = f"{_num_summary(before)} -> {_num_summary(after)}"
        if require_exercised:
            res.append(CheckResult("cache_exercised", advanced, delta))
        else:
            note = ("advanced" if advanced else
                    "no counter moved - APC bypassed under quantized KV "
                    "(mlx-vlm has no QuantizedKVCache APC support); graceful no-op")
            res.append(CheckResult("cache_exercised_info", True, f"{note}; {delta}"))
        return res
    return _check


def pc_disk_cache_created(path: str) -> Callable:
    def _check(client):
        exists = os.path.isdir(os.path.expanduser(path))
        nonempty = exists and any(os.scandir(os.path.expanduser(path)))
        return [CheckResult("disk_cache_dir", exists and nonempty,
                            f"{path}: exists={exists} nonempty={nonempty}")]
    return _check


def pc_residency_budget(pinned_id: str, max_resident: int) -> Callable:
    def _check(client):
        res = _resident(client)
        ids = [i for e in res for i in e.get("ids", [])]
        pinned_ok = pinned_id in ids
        count_ok = len(res) <= max_resident
        return [CheckResult("pinned_resident", pinned_ok,
                            f"pinned={pinned_id} resident_ids={ids}"),
                CheckResult("lru_evicted", count_ok,
                            f"{len(res)} resident <= {max_resident}")]
    return _check


def pc_ttl_reaped(nonpinned_id: str, pinned_id: str, model_field_nonpinned: str,
                  ttl_s: float, wait_s: float) -> Callable:
    def _check(client):
        # touch the non-pinned model so it becomes resident
        client.chat(model_field_nonpinned, P.p_capital().messages, max_tokens=8,
                    temperature=0.0)
        deadline = time.monotonic() + wait_s
        reaped = False
        while time.monotonic() < deadline:
            ids = [i for e in _resident(client) for i in e.get("ids", [])]
            if nonpinned_id not in ids and pinned_id in ids:
                reaped = True
                break
            time.sleep(1.0)
        ids = [i for e in _resident(client) for i in e.get("ids", [])]
        return [CheckResult("ttl_reaped_nonpinned", reaped,
                            f"after {wait_s:.0f}s resident={ids} (ttl={ttl_s})"),
                CheckResult("ttl_kept_pinned", pinned_id in ids,
                            f"pinned {pinned_id} resident={ids}")]
    return _check


def pc_unload_then_reload(model_field: str) -> Callable:
    def _check(client):
        # ensure resident, then unload just this id, then re-request (reloads)
        client.chat(model_field, P.p_capital().messages, max_tokens=8, temperature=0.0)
        before = len(_resident(client))
        st, body = client.unload(model_field)
        after = len(_resident(client))
        res = [CheckResult("unload_one_status", st == 200, f"status={st}"),
               CheckResult("unload_one_shrinks", after < before or before == 0,
                           f"{before} -> {after}")]
        st2, b2 = client.chat(model_field, P.p_capital().messages, max_tokens=8,
                              temperature=0.0)
        res.append(CheckResult("reload_after_unload", st2 == 200, f"status={st2}"))
        return res
    return _check


def pc_reload_endpoint() -> Callable:
    def _check(client):
        st, body = client.reload()
        return [CheckResult("reload_endpoint", st == 200, f"status={st}")]
    return _check


def pc_cache_reset() -> Callable:
    def _check(client):
        st, body = client.cache_reset()
        return [CheckResult("cache_reset", st == 200, f"status={st}")]
    return _check


def pc_hf_gate(stray_id: str) -> Callable:
    def _check(client):
        st, body = client.chat(stray_id, P.p_capital().messages, max_tokens=8,
                               temperature=0.0)
        # must be refused (any non-200) and not silently served
        return [CheckResult("hf_gate_refuses", st != 200,
                            f"stray id status={st} body={str(body)[:120]}")]
    return _check


def pc_metrics_resident_shape() -> Callable:
    def _check(client):
        st, body = client.metrics()
        ok = (st == 200 and isinstance(body, dict)
              and isinstance((body.get("server") or {}).get("resident_models"), list))
        return [CheckResult("metrics_resident_view", ok, f"status={st}")]
    return _check


def _num_summary(stats) -> str:
    if not isinstance(stats, dict):
        return "n/a"
    nums = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    return ", ".join(f"{k}={v}" for k, v in sorted(nums.items())) or "no-counters"


def _stats_advanced(before, after) -> bool:
    if not (isinstance(before, dict) and isinstance(after, dict)):
        return False
    for k, v in after.items():
        if isinstance(v, (int, float)) and isinstance(before.get(k), (int, float)):
            if v > before[k]:
                return True
    return False


# config-dict helpers
def _model_entry(path: str, **kw) -> dict:
    e = {"path": path}
    e.update(kw)
    return e


# scenario builders (one per feature/combination)
def build_scenarios(reg, *, tiers, tmpdir: str, image_path: Optional[str],
                    quick: bool = False) -> list:
    """Construct every scenario whose tier is selected and whose models exist."""
    out: list = []
    suite = P.quick_suite if quick else P.core_suite

    def add(s: Scenario):
        if s.tier in tiers and not reg.missing(*s.needs):
            out.append(s)

    qwen4 = reg.find("qwen3_0_6b_q4")
    qwen8 = reg.find("qwen3_0_6b_q8")
    g1b = reg.find("gemma3_1b")
    e2b = reg.find("gemma4_e2b")
    mmproj = reg.find("gemma4_e2b_mmproj")
    assistant = reg.find("gemma4_e2b_assistant")

    # core: single positional
    # The coherence-judged tiers run on gemma-4-E2B: a 0.6B model is too weak to
    # follow strict-format prompts, so the judge correctly fails it on baseline
    # weakness, drowning out real config regressions. E2B gives a clean baseline
    # where a config-induced degradation actually stands out. The structural tiers
    # (residency/discovery/hf_gate) keep the tiny models - they need distinct small
    # sizes for eviction and only fire the easy `capital` prompt.
    add(Scenario(
        key="single_positional", tier="core", needs=["gemma4_e2b"],
        title="Single positional GGUF (no config)",
        serve_args=[e2b or ""],
        targets=[ReqTarget("default", "", prompts=suite())],
        post=[pc_health_ready(), pc_metrics_resident_shape()],
        notes="baseline: the untouched single-model path still serves coherently"))

    # core: config profiles + precedence + negatives
    cfg_prof = {
        "server": {"defaults": {"profile": "base", "model": "m"}},
        "profiles": {
            "base": {"sampling": {"temperature": 0.0, "max_tokens": 256}},
            "creative": {"extends": "base",
                         "sampling": {"temperature": 1.0, "top_p": 0.98, "min_p": 0.05}},
            "terse": {"extends": "base",
                      "system": "Always answer in a single short sentence."},
        },
        "rules": [{"match": "*coder*", "profile": "terse"}],
        "models": {"m": _model_entry(e2b or "", profile="base")},
        "aliases": {"fast": "m@creative"},
    }
    add(Scenario(
        key="config_profiles", tier="core", needs=["gemma4_e2b"],
        title="Config: profiles, extends, @profile, system, aliases, negatives",
        config=cfg_prof,
        targets=[
            ReqTarget("base", "m", prompts=[P.p_capital(), P.p_instruct()]),
            ReqTarget("inline_creative", "m@creative",
                      prompts=[P.p_capital()], sampling={}),
            ReqTarget("profile_field", "m",
                      prompts=[P.p_capital()], sampling={"profile": "terse"}),
            ReqTarget("system_terse", "m@terse",
                      prompts=[P.p_system_uppercase()]),
            ReqTarget("alias_preset", "fast", prompts=[P.p_capital()]),
            ReqTarget("unknown_id", "no-such-model",
                      prompts=[P.p_capital()], expect_status=404),
            ReqTarget("unknown_profile", "m@nope",
                      prompts=[P.p_capital()], expect_status=400),
        ],
        post=[pc_models_exactly({"m"})],
        notes="precedence ladder + addressing + clean 404/400"))

    # kv: baseline / 8-bit / 4-bit, deep needle recall
    for label, load in (("baseline", {}),
                        ("kv8", {"kv_bits": 8, "kv_group_size": 64}),
                        ("kv4", {"kv_bits": 4, "kv_group_size": 32,
                                 "quantized_kv_start": 0})):
        cfg_kv = {
            "profiles": {"p": {"sampling": {"temperature": 0.0},
                              **({"load": load} if load else {})}},
            "models": {"m": _model_entry(e2b or "", profile="p")},
        }
        add(Scenario(
            key=f"kv_{label}", tier="kv", needs=["gemma4_e2b"],
            title=f"Quantized KV cache: {label} - deep needle recall + long gen",
            config=cfg_kv,
            targets=[ReqTarget("recall", "m",
                               prompts=[P.p_long_ctx_needle(f"VIOLET{label.upper()}88"),
                                        P.p_long_gen()])],
            notes="KV-quant must still recall a planted fact at depth without looping"))

    # cache: disabled / memory / disk / diskxkv8
    # The cache tier runs on Qwen3-0.6B, NOT the judged-tier gemma-4-E2B: APC block /
    # prefix reuse is only supported on a plain ``KVCache`` (mlx-vlm's
    # ``_cache_entry_supports_block_apc``). gemma-4's sliding-window attention uses a
    # ``RotatingKVCache``, so APC is bypassed entirely (zero lookups) and reuse can't
    # be verified at all. To actually exercise the APC feature we need an APC-capable
    # model; the only judged prompt here is the anchored `capital`, so E2B buys nothing.
    add(Scenario(
        key="cache_disabled", tier="cache", needs=["qwen3_0_6b_q4"],
        title="APC prompt cache disabled",
        config={"server": {"cache": {"enabled": False}},
                "models": {"m": _model_entry(qwen4 or "")}},
        targets=[ReqTarget("plain", "m", prompts=[P.p_capital()])],
        post=[pc_apc_enabled(False)],
        notes="apc off path still serves; /health reports apc_enabled=false"))

    add(Scenario(
        key="cache_memory", tier="cache", needs=["qwen3_0_6b_q4"],
        title="APC prompt cache memory-only",
        config={"server": {"cache": {"enabled": True}},
                "models": {"m": _model_entry(qwen4 or "")}},
        targets=[ReqTarget("warm", "m", prompts=[P.p_capital()])],
        post=[pc_apc_enabled(True),
              pc_cache_reuse("m", P.p_long_gen())],
        notes="cache reuse must be byte-identical (no corruption) + counter advances"))

    disk_dir = os.path.join(tmpdir, "apc_disk")
    add(Scenario(
        key="cache_disk", tier="cache", needs=["qwen3_0_6b_q4"],
        title="APC prompt cache with SSD disk tier",
        config={"server": {"cache": {"enabled": True,
                                     "disk": {"path": disk_dir, "max_gb": 2}}},
                "models": {"m": _model_entry(qwen4 or "")}},
        targets=[ReqTarget("warm", "m", prompts=[P.p_capital()])],
        post=[pc_apc_enabled(True),
              pc_cache_reuse("m", P.p_long_gen()),
              pc_disk_cache_created(disk_dir)],
        notes="disk tier created + reused without corrupting output"))

    disk_dir_kv = os.path.join(tmpdir, "apc_disk_kv8")
    add(Scenario(
        key="cache_disk_kv8", tier="cache", needs=["qwen3_0_6b_q8"],
        title="Combination: SSD disk cache x 8-bit quantized KV",
        config={"server": {"cache": {"enabled": True,
                                     "disk": {"path": disk_dir_kv, "max_gb": 2}}},
                "profiles": {"p": {"sampling": {"temperature": 0.0},
                                  "load": {"kv_bits": 8, "kv_group_size": 64}}},
                "models": {"m": _model_entry(qwen8 or "", profile="p")}},
        targets=[ReqTarget("warm_recall", "m",
                           prompts=[P.p_long_ctx_needle("TEALKV8RUN")])],
        post=[pc_cache_reuse("m", P.p_long_ctx_needle("CACHEDNEEDLE9"),
                             require_exercised=False),
              pc_disk_cache_created(disk_dir_kv)],
        notes="quantized-KV + disk-cache co-enabled: mlx-vlm bypasses APC under a "
              "QuantizedKVCache, so this asserts graceful degradation (no crash, no "
              "corruption, disk tier created), not cache reuse"))

    # residency: LRU eviction + idle TTL
    if reg.have("qwen3_0_6b_q4", "qwen3_0_6b_q8", "gemma3_1b"):
        # budget so exactly the two smallest fit -> the third forces an eviction
        sizes = {"q4": os.path.getsize(qwen4), "q8": os.path.getsize(qwen8),
                 "g1b": os.path.getsize(g1b)}
        two_smallest = sorted(sizes.values())[:2]
        budget_gb = round((sum(two_smallest) + 64 * 1024**2) / 1024**3, 3)
        add(Scenario(
            key="residency_lru", tier="residency",
            needs=["qwen3_0_6b_q4", "qwen3_0_6b_q8", "gemma3_1b"],
            title="Multi-model LRU eviction under a tight weight-byte budget",
            serve_args=["--budget-gb", str(budget_gb)],
            config={"models": {
                "q8": _model_entry(qwen8, pin=True),
                "q4": _model_entry(qwen4),
                "g1b": _model_entry(g1b)}},
            targets=[ReqTarget("touch_q8", "q8", prompts=[P.p_capital()]),
                     ReqTarget("touch_q4", "q4", prompts=[P.p_capital()]),
                     ReqTarget("touch_g1b", "g1b", prompts=[P.p_capital()])],
            post=[pc_residency_budget("q8", max_resident=2),
                  pc_models_exactly({"q8", "q4", "g1b"})],
            notes="pinned q8 stays resident; budget forces LRU eviction of a third"))

    if reg.have("qwen3_0_6b_q4", "gemma3_1b"):
        add(Scenario(
            key="residency_ttl", tier="residency",
            needs=["qwen3_0_6b_q4", "gemma3_1b"],
            title="Idle-TTL reaper unloads a non-pinned model, keeps the pinned one",
            env={"MLX_VLM_RESIDENT_TTL_TICK": "1"},
            config={"server": {"defaults": {"ttl_s": 4}},
                    "models": {"keep": _model_entry(qwen4, pin=True),
                              "drop": _model_entry(g1b)}},
            targets=[ReqTarget("touch_keep", "keep", prompts=[P.p_capital()])],
            post=[pc_ttl_reaped("drop", "keep", "drop", ttl_s=4, wait_s=20)],
            notes="fast clock (ttl=4s, tick=1s): non-pinned idle model is reaped"))

    # template: chat_template override
    tmpl_path = os.path.join(tmpdir, "custom_template.jinja")
    add(Scenario(
        key="chat_template_config", tier="template", needs=["gemma4_e2b"],
        title="Config chat_template override + distinct-template resident fork",
        config={
            "profiles": {"tmpl": {"chat_template": _SIMPLE_TEMPLATE}},
            "models": {"plain": _model_entry(e2b or ""),
                      "templated": _model_entry(e2b or "", profile="tmpl")}},
        targets=[ReqTarget("plain", "plain", prompts=[P.p_capital()]),
                 ReqTarget("templated", "templated", prompts=[P.p_capital()])],
        post=[_pc_two_resident_entries_same_path(e2b or "")],
        notes="same GGUF under two templates => two resident entries; both coherent"))

    add(Scenario(
        key="chat_template_cli", tier="template", needs=["gemma4_e2b"],
        title="Single-model --chat-template flag",
        serve_args=[e2b or "", "--chat-template", _make_template_file(tmpl_path)],
        targets=[ReqTarget("default", "", prompts=[P.p_capital(), P.p_instruct()])],
        notes="the CLI override rides the same overrides slot and stays coherent"))

    # endpoints: unload / reload / metrics / cache reset
    add(Scenario(
        key="endpoints", tier="endpoints", needs=["gemma4_e2b"],
        title="Operational endpoints: unload(one+all), reload, metrics, cache reset",
        config={"server": {"cache": {"enabled": True}},
                "models": {"m": _model_entry(e2b or "")}},
        targets=[ReqTarget("warm", "m", prompts=[P.p_capital()], stream=True)],
        post=[pc_metrics_resident_shape(),
              pc_cache_reset(),
              pc_unload_then_reload("m"),
              pc_reload_endpoint()],
        notes="also exercises the SSE streaming path on the warm-up request"))

    # negative: HF gate refuses a stray non-GGUF id
    add(Scenario(
        key="hf_gate", tier="negative", needs=["qwen3_0_6b_q4"],
        title="HF-download gate refuses a stray non-GGUF id (offline)",
        config={"server": {"hf_cache": False},
                "models": {"m": _model_entry(qwen4 or "")}},
        targets=[ReqTarget("ok", "m", prompts=[P.p_capital()])],
        post=[pc_hf_gate("mlx-community/Qwen2.5-7B-Instruct-4bit"),
              pc_models_exactly({"m"})],
        notes="no network; an unconfigured repo id is refused, never downloaded"))

    # discovery: --models-dir header scan
    if reg.have("qwen3_0_6b_q4", "gemma3_1b"):
        disc_dir = _make_discovery_dir(os.path.join(tmpdir, "discover"),
                                       [qwen4, g1b])
        add(Scenario(
            key="discovery", tier="discovery",
            needs=["qwen3_0_6b_q4", "gemma3_1b"],
            title="Discovery mode (--models-dir) serves header-derived ids",
            serve_args=["--models-dir", disc_dir, "--recursive"],
            targets=[ReqTarget("first", "__first__", prompts=[P.p_capital()])],
            post=[pc_metrics_resident_shape()],
            notes="a curated dir is scanned; the first derived id serves coherently"))

    # vlm: image description
    if reg.have("gemma4_e2b", "gemma4_e2b_mmproj") and image_path:
        add(Scenario(
            key="vlm_image", tier="vlm", needs=["gemma4_e2b", "gemma4_e2b_mmproj"],
            title="VLM: gemma-4-E2B + mmproj describes an image",
            serve_args=[e2b, "--mmproj", mmproj],
            targets=[ReqTarget("describe", "", prompts=[P.p_vlm_describe()],
                               image_handle=image_path)],
            notes="vision load/serve path under the harness; judge rates the caption"))

    # mtp: speculative + lossless-greedy vs base
    if reg.have("gemma4_e2b", "gemma4_e2b_assistant"):
        add(Scenario(
            key="mtp_speculative", tier="mtp",
            needs=["gemma4_e2b", "gemma4_e2b_assistant"],
            title="MTP: gemma-4-E2B + assistant drafter (speculative) stays coherent",
            config={"models": {"spec": _model_entry(
                e2b, draft_gguf=assistant, speculative=True)}},
            targets=[ReqTarget("spec", "spec",
                               prompts=[P.p_capital(), P.p_instruct(), P.p_long_gen()])],
            notes="E2B assistant-shape MTP may be untested - surface any surprise"))
        add(Scenario(
            key="mtp_lossless", tier="mtp",
            needs=["gemma4_e2b", "gemma4_e2b_assistant"],
            title="MTP lossless-greedy: speculative output == base output",
            config={"models": {
                "spec": _model_entry(e2b, draft_gguf=assistant, speculative=True),
                "base": _model_entry(e2b)}},
            targets=[],     # the comparison is the post-check
            post=[_pc_mtp_lossless("spec", "base")],
            notes="greedy spec vs greedy base, token-for-token; mismatch is a finding"))

    return out


# scenario-specific post-checks that need two requests
def _pc_two_resident_entries_same_path(path: str) -> Callable:
    def _check(client):
        # touch both ids, then assert the pool has two entries backing one path
        client.chat("plain", P.p_capital().messages, max_tokens=8, temperature=0.0)
        client.chat("templated", P.p_capital().messages, max_tokens=8, temperature=0.0)
        res = _resident(client)
        same_path = [e for e in res
                     if os.path.abspath(e.get("model_path", "")) == os.path.abspath(path)]
        ok = len(same_path) >= 2
        return [CheckResult("template_forks_entry", ok,
                            f"{len(same_path)} resident entries for the GGUF "
                            f"(expect >=2: distinct chat templates)")]
    return _check


def _pc_mtp_lossless(spec_id: str, base_id: str) -> Callable:
    def _check(client):
        out = []
        for pr in (P.p_capital(), P.p_instruct(), P.p_count()):
            _, b_spec = client.chat(spec_id, pr.messages, max_tokens=pr.max_tokens,
                                    temperature=0.0)
            _, b_base = client.chat(base_id, pr.messages, max_tokens=pr.max_tokens,
                                    temperature=0.0)
            t_spec = checks.extract_chat_text(b_spec) if isinstance(b_spec, dict) else None
            t_base = checks.extract_chat_text(b_base) if isinstance(b_base, dict) else None
            out.append(CheckResult(f"lossless[{pr.key}]", bool(t_base) and t_spec == t_base,
                                   "identical" if t_spec == t_base
                                   else f"DIVERGED spec={str(t_spec)[:60]!r} "
                                        f"base={str(t_base)[:60]!r}"))
        # Anti-vacuous guard: the comparison is only meaningful if `base` loaded
        # NON-speculatively. One GGUF backs both ids; if the path-keyed MTP registry
        # leaked the drafter into base's build (the worker-thread bug class), base
        # would also be MTP and spec==base would pass for the wrong reason. The MTP
        # target wrapper logs exactly once per speculative build, so the log must
        # carry exactly one - proving base is the bare base.
        n_mtp = _count_in_log(getattr(client, "log_path", None),
                              "(MTP target wrapper)")
        out.append(CheckResult(
            "provenance:base_is_non_spec", n_mtp == 1,
            f"{n_mtp} MTP target build(s) (expect 1: only {spec_id!r}, not {base_id!r})"))
        return out
    return _check


def _count_in_log(log_path, needle) -> int:
    """Occurrences of ``needle`` in the live server log (builds print to it as they
    happen). -1 if the log isn't readable, so the guard fails loud rather than green."""
    if not log_path:
        return -1
    try:
        with open(log_path) as f:
            return f.read().count(needle)
    except OSError:
        return -1


# fixtures written to the temp dir
_SIMPLE_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}"
    "<|assistant|>\n")


def _make_template_file(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_SIMPLE_TEMPLATE)
    return path


def _make_discovery_dir(path: str, ggufs: list) -> str:
    """A curated discovery dir: symlink a couple of small GGUFs so --models-dir
    scans a controlled set (not the user's whole library)."""
    os.makedirs(path, exist_ok=True)
    for g in ggufs:
        if not g:
            continue
        link = os.path.join(path, os.path.basename(g))
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(g), link)
    return path


ALL_TIERS = ("core", "kv", "cache", "residency", "template", "endpoints",
             "negative", "discovery", "vlm", "mtp")
