"""Result model + Markdown/JSON report for the server e2e harness.

The runner fills :class:`ScenarioResult` / :class:`Transcript` objects as it drives
each server; this module renders them two ways - a human-readable Markdown summary
and a machine-diffable JSON dump - mirroring the lab's "report + JSON" house style so
runs are both readable and programmatically comparable across commits.

Pure stdlib, no model imports, so it's usable from ``--dry-run`` too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# result model (shared by the runner)
@dataclass
class Transcript:
    """One request/response, with its floor checks, anchor, and (later) judge verdict."""
    scenario_key: str
    tier: str
    target: str
    prompt_key: str
    kind: str
    model_field: str
    request_summary: str
    status: int = 0
    expect_status: int = 200
    text: str = ""
    floor: list = field(default_factory=list)     # list[CheckResult-as-dict]
    anchor: Optional[dict] = None                 # CheckResult-as-dict or None
    judge: Optional[dict] = None                  # Verdict.to_dict() or None
    judge_eligible: bool = False
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def status_ok(self) -> bool:
        return self.status == self.expect_status

    @property
    def floor_ok(self) -> bool:
        # floor only matters for a response we expected to succeed
        if self.expect_status != 200:
            return True
        return all(r.get("ok") for r in self.floor)

    @property
    def anchor_ok(self) -> bool:
        return self.anchor is None or bool(self.anchor.get("ok"))

    @property
    def judge_ok(self) -> bool:
        return self.judge is None or bool(self.judge.get("ok"))

    @property
    def ok(self) -> bool:
        return (self.status_ok and self.floor_ok and self.anchor_ok
                and self.judge_ok and self.error is None)


@dataclass
class ScenarioResult:
    key: str
    title: str
    tier: str
    skipped: bool = False
    skip_reason: str = ""
    launched: bool = False
    launch_error: Optional[str] = None
    serve_args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    transcripts: list = field(default_factory=list)     # list[Transcript]
    post: list = field(default_factory=list)            # list[CheckResult-as-dict]
    notes: str = ""

    @property
    def post_ok(self) -> bool:
        return all(r.get("ok") for r in self.post)

    @property
    def ok(self) -> bool:
        if self.skipped:
            return True
        if not self.launched:
            return False
        return self.post_ok and all(t.ok for t in self.transcripts)


# rendering
def _check_line(r: dict) -> str:
    mark = "ok" if r.get("ok") else "FAIL"
    detail = r.get("detail") or ""
    return f"  - [{mark}] {r.get('name')}: {detail}".rstrip()


def _status_icon(ok: bool, skipped: bool = False) -> str:
    if skipped:
        return "SKIP"
    return "PASS" if ok else "FAIL"


def render_markdown(results: list, *, meta: dict) -> str:
    lines: list = []
    a = lines.append
    a("# gmlx server e2e report")
    a("")
    a(f"- generated: {meta.get('timestamp', 'n/a')}")
    a(f"- tiers: {', '.join(meta.get('tiers', []))}")
    a(f"- models root: {meta.get('models_root')}")
    a(f"- judge: {meta.get('judge') or '(none)'}")
    a(f"- mode: {meta.get('mode', 'run')}")
    a("")

    ran = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    passed = [r for r in ran if r.ok]
    failed = [r for r in ran if not r.ok]

    a("## Summary")
    a("")
    a(f"- scenarios: {len(results)} ({len(ran)} ran, {len(skipped)} skipped)")
    a(f"- passed: {len(passed)}")
    a(f"- failed: {len(failed)}")
    n_req = sum(len(r.transcripts) for r in ran)
    n_req_fail = sum(1 for r in ran for t in r.transcripts if not t.ok)
    a(f"- requests: {n_req} ({n_req_fail} failing)")
    a("")

    # per-tier roll-up
    tiers: dict = {}
    for r in ran:
        t = tiers.setdefault(r.tier, [0, 0])
        t[0] += 1
        t[1] += 1 if r.ok else 0
    if tiers:
        a("| tier | scenarios | passed |")
        a("| --- | --- | --- |")
        for tier in sorted(tiers):
            tot, ok = tiers[tier]
            a(f"| {tier} | {tot} | {ok} |")
        a("")

    if failed:
        a("## Failures")
        a("")
        for r in failed:
            a(f"- **{r.key}** ({r.tier}) - {r.title}")
            if r.launch_error:
                a(f"  - launch error: {r.launch_error}")
            for pr in r.post:
                if not pr.get("ok"):
                    a(_check_line(pr).replace("  - ", "  - post "))
            for t in r.transcripts:
                if not t.ok:
                    a(f"  - request `{t.target}/{t.prompt_key}` "
                      f"(status {t.status}, want {t.expect_status})")
                    if t.error:
                        a(f"    - error: {t.error}")
                    for fr in t.floor:
                        if not fr.get("ok"):
                            a("    " + _check_line(fr).strip())
                    if t.anchor and not t.anchor.get("ok"):
                        a("    " + _check_line(t.anchor).strip())
                    if t.judge and not t.judge.get("ok"):
                        a(f"    - judge: score={t.judge.get('score')} "
                          f"coherent={t.judge.get('coherent')} "
                          f"rep={t.judge.get('repetition')} "
                          f"- {t.judge.get('reason')}")
        a("")

    a("## Scenarios")
    a("")
    for r in results:
        icon = _status_icon(r.ok, r.skipped)
        a(f"### [{icon}] {r.key} - {r.title}")
        a(f"_tier: {r.tier}_")
        if r.skipped:
            a(f"- skipped: {r.skip_reason}")
            a("")
            continue
        if r.config_path:
            a(f"- config: `{r.config_path}`")
        if r.serve_args:
            a(f"- serve args: `{' '.join(str(x) for x in r.serve_args)}`")
        if r.env:
            a(f"- env: `{', '.join(f'{k}={v}' for k, v in r.env.items())}`")
        if r.log_path:
            a(f"- server log: `{r.log_path}`")
        if not r.launched:
            a(f"- **did not launch**: {r.launch_error}")
            a("")
            continue
        if r.post:
            a("- post-checks:")
            for pr in r.post:
                a(_check_line(pr))
        for t in r.transcripts:
            tick = "ok" if t.ok else "FAIL"
            a(f"- request **{t.target}/{t.prompt_key}** [{tick}] "
              f"(status {t.status}, {t.elapsed_s:.1f}s)")
            if t.error:
                a(f"  - error: {t.error}")
            failing = [fr for fr in t.floor if not fr.get("ok")]
            for fr in failing:
                a("  " + _check_line(fr).strip())
            if t.anchor:
                a("  " + _check_line(t.anchor).strip())
            if t.judge:
                jmark = "ok" if t.judge.get("ok") else "FAIL"
                a(f"  - [judge {jmark}] score={t.judge.get('score')} "
                  f"coherent={t.judge.get('coherent')} "
                  f"rep={t.judge.get('repetition')} parsed={t.judge.get('parsed')} "
                  f"- {t.judge.get('reason')}")
            snippet = (t.text or "").strip().replace("\n", " ")
            if snippet:
                a(f"  - text: {snippet[:200]!r}")
        a("")
    return "\n".join(lines) + "\n"


def to_json(results: list, *, meta: dict) -> dict:
    def _t(t: Transcript) -> dict:
        return {
            "target": t.target, "prompt_key": t.prompt_key, "kind": t.kind,
            "model_field": t.model_field, "request_summary": t.request_summary,
            "status": t.status, "expect_status": t.expect_status,
            "status_ok": t.status_ok, "floor_ok": t.floor_ok,
            "anchor_ok": t.anchor_ok, "judge_ok": t.judge_ok, "ok": t.ok,
            "floor": t.floor, "anchor": t.anchor, "judge": t.judge,
            "judge_eligible": t.judge_eligible, "elapsed_s": round(t.elapsed_s, 3),
            "error": t.error, "text": (t.text or "")[:2000],
        }

    def _s(r: ScenarioResult) -> dict:
        return {
            "key": r.key, "title": r.title, "tier": r.tier,
            "skipped": r.skipped, "skip_reason": r.skip_reason,
            "launched": r.launched, "launch_error": r.launch_error,
            "ok": r.ok, "post_ok": r.post_ok,
            "serve_args": [str(x) for x in r.serve_args], "env": r.env,
            "config_path": r.config_path, "log_path": r.log_path,
            "notes": r.notes, "post": r.post,
            "transcripts": [_t(t) for t in r.transcripts],
        }

    ran = [r for r in results if not r.skipped]
    return {
        "meta": meta,
        "summary": {
            "scenarios": len(results),
            "ran": len(ran),
            "skipped": sum(1 for r in results if r.skipped),
            "passed": sum(1 for r in ran if r.ok),
            "failed": sum(1 for r in ran if not r.ok),
            "requests": sum(len(r.transcripts) for r in ran),
            "requests_failed": sum(1 for r in ran for t in r.transcripts if not t.ok),
        },
        "scenarios": [_s(r) for r in results],
    }


def write_report(results: list, *, meta: dict, md_path: str, json_path: str) -> None:
    with open(md_path, "w") as f:
        f.write(render_markdown(results, meta=meta))
    with open(json_path, "w") as f:
        json.dump(to_json(results, meta=meta), f, indent=2)
        f.write("\n")
