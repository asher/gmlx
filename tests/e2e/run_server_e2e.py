#!/usr/bin/env python3
"""End-to-end test driver for the gmlx server.

Launches the real server (``python -m gmlx.server``) across a matrix of start
modes and config features, fires a fixed prompt suite plus per-feature post-checks at
each, and grades every response two ways: deterministic *floor* checks (transport,
schema, repetition, mojibake, NaN - ``checks.py``) and a decoupled LLM-as-judge pass
for semantic coherence (``judge.py``). Emits a Markdown + JSON report.

Phases:
  0. build + validate (CPU)   registry, scenario matrix, YAML round-trip - ``--dry-run``
                              stops here, loading no model.
  1. run scenarios (GPU)      launch each server, run targets + post-checks, tear down.
  2. judge (GPU)              score eligible transcripts against a local judge model,
                              after every server is down (no GPU contention).
  3. report                   write report.md / report.json under ``--out``.

Run with the project interpreter::

  python tests/e2e/run_server_e2e.py --dry-run            # CPU, validate the matrix
  python tests/e2e/run_server_e2e.py --tiers core,kv      # GPU, a subset
  python tests/e2e/run_server_e2e.py --judge-only OUT/report.json   # re-grade only

This file is intentionally *not* ``test_``-prefixed so pytest does not collect it (it
drives real servers + GPU). The sibling modules import each other by bare name, so we
prepend this directory to ``sys.path`` before importing them.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml                                    # noqa: E402  (after sys.path shim)

import checks                                  # noqa: E402
import report as R                             # noqa: E402
import scenarios as SC                         # noqa: E402
from client import Client                      # noqa: E402
from models import DEFAULT_ROOT, ModelRegistry  # noqa: E402
from server_proc import ServerLaunchError, ServerProc  # noqa: E402


# small helpers
def _cr(c) -> dict:
    return {"name": c.name, "ok": bool(c.ok), "detail": c.detail}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _dump_config(cfg: dict, path: str) -> str:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def _ensure_image(out_dir: str, image_arg) -> "str | None":
    """Resolve an image for the VLM tier. Order of preference:

      1. an explicit ``--image PATH``
      2. ``$GMLX_E2E_IMAGE``
      3. the first image bundled under ``tests/e2e/assets/``

    A real photo is strongly preferred - it actually exercises the vision encoder.
    Only if none of the above resolve do we synthesize a trivial shapes PNG as a
    last-resort backstop (it proves the load/serve path works but is a weak vision
    test). Returns None if even that can't be produced, in which case the VLM-image
    scenario is skipped.
    """
    # 1 + 2: explicit path, then env var
    for cand in (image_arg, os.environ.get("GMLX_E2E_IMAGE")):
        if cand:
            p = os.path.abspath(os.path.expanduser(cand))
            if os.path.exists(p):
                return p
    # 3: a bundled asset, if one ships with the harness
    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    if os.path.isdir(assets):
        for name in sorted(os.listdir(assets)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return os.path.join(assets, name)
    # last resort: a synthetic backstop (weak vision test)
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    path = os.path.join(out_dir, "synthetic.png")
    img = Image.new("RGB", (224, 224), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.ellipse((48, 48, 176, 176), fill=(200, 40, 40))
    d.rectangle((20, 20, 60, 60), fill=(30, 120, 200))
    img.save(path)
    return path


# phase 0 - build + validate
def build_matrix(reg, *, tiers, out_dir, image_path, quick, filt):
    fixtures = os.path.join(out_dir, "fixtures")
    os.makedirs(fixtures, exist_ok=True)
    scenarios = SC.build_scenarios(reg, tiers=tiers, tmpdir=fixtures,
                                   image_path=image_path, quick=quick)
    if filt:
        scenarios = [s for s in scenarios if filt in s.key]
    return scenarios


def validate_configs(scenarios, out_dir) -> list:
    """Round-trip every scenario config through the real loader (build_config +
    load_config from YAML). Returns a list of (key, error) for any that fail."""
    from gmlx.config import build_config, load_config       # CPU-only import
    cfg_dir = os.path.join(out_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    errors = []
    for s in scenarios:
        if s.config is None:
            continue
        path = os.path.join(cfg_dir, f"{s.key}.yaml")
        try:
            build_config(s.config)            # validates the in-memory dict
            _dump_config(s.config, path)
            load_config(path)                 # validates the dumped YAML
        except Exception as e:                # noqa: BLE001
            errors.append((s.key, f"{type(e).__name__}: {e}"))
    return errors


# phase 1 - run a scenario against a live server
def _resolve_model_field(client: Client, target) -> str:
    if target.model == "__first__":
        st, body = client.models()
        if st == 200 and isinstance(body, dict) and body.get("data"):
            return body["data"][0]["id"]
        return ""                              # fall back to the server default
    return target.model                        # "" => default; id / id@profile pass through


def _run_request(client, s, tgt, pr, model_field, image_path) -> R.Transcript:
    images = None
    if "vlm" in pr.needs:
        handle = tgt.image_handle or image_path
        images = [handle] if handle else None
    tr = R.Transcript(
        scenario_key=s.key, tier=s.tier, target=tgt.name, prompt_key=pr.key,
        kind=pr.kind, model_field=model_field, request_summary=pr.request_summary(),
        expect_status=tgt.expect_status)
    t0 = time.monotonic()
    # The judged tiers run on thinking-capable models (gemma-4-E2B, Qwen3); a
    # reasoning preamble would pollute the judged text and the floor checks. Disable
    # thinking by default (a target can re-enable via sampling) so we judge the answer.
    sampling = {"enable_thinking": False, **tgt.sampling}
    try:
        status, body = client.chat(model_field, pr.messages, max_tokens=pr.max_tokens,
                                   stream=tgt.stream, image_paths=images, **sampling)
    except Exception as e:                     # noqa: BLE001
        tr.error = f"{type(e).__name__}: {e}"
        tr.elapsed_s = time.monotonic() - t0
        return tr
    tr.elapsed_s = time.monotonic() - t0
    tr.status = status if isinstance(status, int) else 0

    if tgt.expect_status != 200:               # negative test: only the status matters
        return tr

    bdict = body if isinstance(body, dict) else {}
    floor = checks.floor_text_checks(tr.status, bdict)
    tr.floor = [_cr(c) for c in floor.results]
    text = checks.extract_chat_text(bdict) if bdict else None
    tr.text = text or ""

    if pr.kind == "system" and pr.key == "system_uppercase":
        frac = checks.fraction_uppercase_letters(text or "")
        tr.anchor = {"name": "uppercase", "ok": frac > 0.9,
                     "detail": f"{frac:.0%} uppercase"}
    elif pr.anchors:
        cr = checks.check_contains(text or "", pr.anchors["substrs"],
                                   mode=pr.anchors.get("mode", "any"), name="anchor")
        tr.anchor = _cr(cr)

    tr.judge_eligible = bool(pr.judge)
    return tr


def run_scenario(s, *, out_dir, image_path, python) -> R.ScenarioResult:
    res = R.ScenarioResult(key=s.key, title=s.title, tier=s.tier, notes=s.notes,
                           env=dict(s.env))
    cfg_path = None
    if s.config is not None:
        cfg_path = os.path.join(out_dir, "configs", f"{s.key}.yaml")
        _dump_config(s.config, cfg_path)
    serve_args = (["--config", cfg_path] if cfg_path else []) + list(s.serve_args)
    res.config_path = cfg_path
    res.serve_args = serve_args
    log_path = os.path.join(out_dir, "logs", f"{s.key}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    res.log_path = log_path

    proc = ServerProc(serve_args, env_extra=s.env, log_path=log_path, python=python)
    try:
        proc.start()
        proc.wait_ready()
        res.launched = True
        client = Client(proc.base_url)
        for tgt in s.targets:
            model_field = _resolve_model_field(client, tgt)
            for pr in tgt.prompts:
                for _ in range(max(1, tgt.repeat)):
                    res.transcripts.append(
                        _run_request(client, s, tgt, pr, model_field, image_path))
        client.log_path = res.log_path     # post-checks that inspect build provenance
        for pc in s.post:
            try:
                for cr in pc(client):
                    res.post.append(_cr(cr))
            except Exception as e:             # noqa: BLE001
                res.post.append({"name": "post_error", "ok": False,
                                 "detail": f"{type(e).__name__}: {e}"})
    except ServerLaunchError as e:
        res.launch_error = str(e)
    except Exception as e:                     # noqa: BLE001
        res.launch_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        proc.stop()
    return res


# phase 2 - judge
def run_judge(results, *, judge_path, hf_source, min_score) -> None:
    from judge import Judge
    j = Judge(judge_path, hf_source=hf_source, min_score=min_score)
    n = 0
    for r in results:
        for t in r.transcripts:
            if t.judge_eligible and t.status == 200 and (t.text or "").strip():
                v = j.score(t.request_summary, t.text)
                t.judge = v.to_dict()
                n += 1
    print(f"[judge] scored {n} response(s) with {os.path.basename(judge_path)}")


# judge-only - reconstruct results from a prior JSON report
def _results_from_json(path) -> tuple:
    import json
    with open(path) as f:
        data = json.load(f)
    out = []
    for s in data.get("scenarios", []):
        r = R.ScenarioResult(
            key=s["key"], title=s["title"], tier=s["tier"], skipped=s["skipped"],
            skip_reason=s.get("skip_reason", ""), launched=s["launched"],
            launch_error=s.get("launch_error"), serve_args=s.get("serve_args", []),
            env=s.get("env", {}), config_path=s.get("config_path"),
            log_path=s.get("log_path"), notes=s.get("notes", ""), post=s.get("post", []))
        for t in s.get("transcripts", []):
            r.transcripts.append(R.Transcript(
                scenario_key=s["key"], tier=s["tier"], target=t["target"],
                prompt_key=t["prompt_key"], kind=t["kind"],
                model_field=t["model_field"], request_summary=t["request_summary"],
                status=t["status"], expect_status=t["expect_status"], text=t["text"],
                floor=t["floor"], anchor=t["anchor"], judge=t.get("judge"),
                judge_eligible=t["judge_eligible"], elapsed_s=t["elapsed_s"],
                error=t.get("error")))
        out.append(r)
    return out, data.get("meta", {})


# plan printing
def print_plan(reg, scenarios, *, tiers, image_path) -> None:
    inv = reg.inventory()
    print(f"models root: {reg._root()}")
    print("inventory:")
    for h, p in inv.items():
        print(f"  {'[x]' if p else '[ ]'} {h}: {p or '(missing)'}")
    print(f"image: {image_path or '(none - vlm-image scenarios skip)'}")
    print(f"tiers selected: {', '.join(tiers)}")
    print(f"scenarios constructed: {len(scenarios)}")
    by_tier = {}
    for s in scenarios:
        by_tier.setdefault(s.tier, []).append(s.key)
    for tier in sorted(by_tier):
        print(f"  {tier}: {', '.join(by_tier[tier])}")
    # which selected tiers produced nothing (models missing)
    empty = [t for t in tiers if t not in by_tier]
    if empty:
        print(f"tiers with no runnable scenario (models missing): {', '.join(empty)}")


# main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_server_e2e",
        description="End-to-end config-matrix test harness for the gmlx server.")
    ap.add_argument("--tiers", default="all",
                    help=f"Comma list of tiers (or 'all'). Known: {', '.join(SC.ALL_TIERS)}")
    ap.add_argument("--filter", default=None,
                    help="Only run scenarios whose key contains this substring.")
    ap.add_argument("--models-root", default=DEFAULT_ROOT,
                    help="Root directory holding the GGUF models (default %(default)s).")
    ap.add_argument("--judge-model", default=None,
                    help="GGUF for the LLM-as-judge (default: best available local).")
    ap.add_argument("--judge-hf-source", default=None,
                    help="Optional processor/config source for the judge model.")
    ap.add_argument("--min-score", type=int, default=3,
                    help="Judge score below this fails a response (1-5; default 3).")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip the LLM-judge phase (floor checks only).")
    ap.add_argument("--judge-only", default=None, metavar="REPORT.json",
                    help="Re-grade a prior run's JSON report with the judge; no servers.")
    ap.add_argument("--quick", action="store_true",
                    help="Use the short prompt suite for the generic core target.")
    ap.add_argument("--image", default=None,
                    help="Image for the VLM tier (a real photo is best). Falls back "
                         "to $GMLX_E2E_IMAGE, then a bundled asset, then a weak "
                         "synthesized PNG.")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: a fresh temp dir).")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter to launch the server subprocess (default: this one).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build + validate the matrix only (CPU); load no model.")
    ap.add_argument("--list", action="store_true",
                    help="Print the plan (inventory + scenarios) and exit.")
    ap.add_argument("--print-pull", action="store_true",
                    help="Print `gmlx pull` commands for any models not yet on "
                         "disk under --models-root, then exit (CPU, no network).")
    a = ap.parse_args(argv)

    # Line-buffer our own stdout/stderr so `tail -f console.log` shows live progress
    # even when stdout is a redirected file (block-buffered by default).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    # print-pull: copy-paste model bootstrap, then stop (CPU, no network)
    if a.print_pull:
        ModelRegistry(root=a.models_root).print_bootstrap()
        return 0

    tiers = (list(SC.ALL_TIERS) if a.tiers == "all"
             else [t.strip() for t in a.tiers.split(",") if t.strip()])
    unknown = [t for t in tiers if t not in SC.ALL_TIERS]
    if unknown:
        ap.error(f"unknown tier(s): {', '.join(unknown)}; known: {', '.join(SC.ALL_TIERS)}")

    out_dir = a.out or tempfile.mkdtemp(prefix=f"gmlx-e2e-{_stamp()}-")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")
    json_path = os.path.join(out_dir, "report.json")

    # judge-only: re-grade an existing report, then stop
    if a.judge_only:
        reg = ModelRegistry(root=a.models_root)
        judge_path = a.judge_model or reg.default_judge()
        if not judge_path:
            print("no judge model available; cannot --judge-only", file=sys.stderr)
            return 2
        results, meta = _results_from_json(a.judge_only)
        run_judge(results, judge_path=judge_path, hf_source=a.judge_hf_source,
                  min_score=a.min_score)
        meta = {**meta, "mode": "judge-only", "judge": judge_path,
                "timestamp": _now()}
        R.write_report(results, meta=meta, md_path=md_path, json_path=json_path)
        print(f"[report] {md_path}")
        return 0 if all(r.ok for r in results) else 1

    # phase 0: build + validate
    reg = ModelRegistry(root=a.models_root)
    image_path = _ensure_image(out_dir, a.image)
    scenarios = build_matrix(reg, tiers=tiers, out_dir=out_dir,
                             image_path=image_path, quick=a.quick, filt=a.filter)
    print_plan(reg, scenarios, tiers=tiers, image_path=image_path)
    cfg_errors = validate_configs(scenarios, out_dir)
    if cfg_errors:
        print("\nCONFIG VALIDATION ERRORS:")
        for key, err in cfg_errors:
            print(f"  {key}: {err}")
        return 2
    print("\n[phase 0] config matrix validated "
          f"({sum(s.config is not None for s in scenarios)} configs round-tripped).")

    if a.list or a.dry_run:
        print(f"[dry-run] no servers launched. Artifacts dir: {out_dir}")
        return 0

    if not scenarios:
        print("no runnable scenarios (no models found under "
              f"{reg._root()}). Nothing to do.")
        return 0

    judge_path = None if a.no_judge else (a.judge_model or reg.default_judge())

    meta = {
        "timestamp": _now(), "tiers": tiers, "models_root": reg._root(),
        "judge": judge_path, "mode": "run", "out_dir": out_dir,
        "mlx_vlm": _mlx_vlm_version(),
    }

    # phase 1: run scenarios
    results = []
    for i, s in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {s.key} ({s.tier}) - {s.title}")
        res = run_scenario(s, out_dir=out_dir, image_path=image_path, python=a.python)
        status = ("launched" if res.launched else "LAUNCH FAILED")
        nfail = sum(1 for t in res.transcripts if not t.ok)
        print(f"    {status}; {len(res.transcripts)} request(s), {nfail} failing; "
              f"post {sum(p['ok'] for p in res.post)}/{len(res.post)}")
        if res.launch_error:
            print("    " + res.launch_error.splitlines()[0])
        results.append(res)
        # interim report so a long run is inspectable as it goes
        R.write_report(results, meta=meta, md_path=md_path, json_path=json_path)

    # phase 2: judge (servers are all down)
    if judge_path:
        print(f"\n[phase 2] judging with {judge_path}")
        try:
            run_judge(results, judge_path=judge_path,
                      hf_source=a.judge_hf_source, min_score=a.min_score)
        except Exception as e:                 # noqa: BLE001
            print(f"[judge] FAILED to run judge: {e}")
            meta["judge_error"] = f"{type(e).__name__}: {e}"
    else:
        print("\n[phase 2] judging skipped")

    # phase 3: report
    R.write_report(results, meta=meta, md_path=md_path, json_path=json_path)
    n_pass = sum(1 for r in results if r.ok)
    print(f"\n[report] {md_path}")
    print(f"[report] {json_path}")
    print(f"RESULT: {n_pass}/{len(results)} scenarios passed")
    return 0 if n_pass == len(results) else 1


def _mlx_vlm_version() -> str:
    try:
        import mlx_vlm                          # noqa: WPS433
        return getattr(mlx_vlm, "__version__", "unknown")
    except Exception:                           # noqa: BLE001
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
