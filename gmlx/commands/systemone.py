"""``gmlx systemone``: send a structured-decision request to a server, or
run it offline on a GGUF: DiffusionGemma, or any text model through the
letter readout."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _print_answers(body: dict) -> None:
    answers = body.get("answers") or {}
    width = max((len(str(k)) for k in answers), default=0)
    for qid, a in answers.items():
        name = f"{qid}:".ljust(width + 1)
        if a is None:
            print(f"{name} skipped")
        elif a.get("type") == "noul":
            print(f"{name} {a['noul']:.3f}")
        elif a.get("type") == "choice":
            print(f"{name} {a['choice']} ({a['confidence']:.2f})")
        else:
            print(f"{name} {a['score']:.2f} ({a['confidence']:.2f})")


def _post(a, body: dict) -> tuple[int, dict | None]:
    if a.url:
        root = a.url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
    else:
        import gmlx.serve.lifecycle as lifecycle

        host, port = lifecycle.auto_target(a.host, a.port)
        root = f"http://{host}:{port}"
    api_key = a.api_key or os.environ.get("GMLX_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(root + "/v1/systemone",
                                 data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as r:  # noqa: S310
            return 0, json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("error: the server requires an API key - pass --api-key "
                  "(or set GMLX_API_KEY)", file=sys.stderr)
            return 1, None
        try:
            err = json.loads(e.read()).get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err)
        except (ValueError, AttributeError):
            message = None
        print(f"error: {root}/v1/systemone returned HTTP {e.code}"
              + (f": {message}" if message else ""), file=sys.stderr)
        return 1, None
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", None) or e
        print(f"error: no gmlx server reachable at {root} ({reason}) - "
              f"start one with `gmlx serve`", file=sys.stderr)
        return 1, None


def _offline_settings(config_path):
    """The ``server.systemone`` section of the CLI config, or its
    defaults when no config is found."""
    import gmlx.config as cfgmod

    try:
        cfg, _path = cfgmod.load_cli_config(config_path)
    except cfgmod.ConfigError as e:
        raise SystemExit(f"error: {e}") from e
    return cfg, (cfg.systemone if cfg is not None else cfgmod.SystemoneCfg())


def _model_path(name: str, cfg) -> str:
    path = os.path.expanduser(name)
    if os.path.exists(path):
        return path
    if cfg is not None and "/" not in name and not name.lower().endswith(".gguf"):
        import gmlx.config as cfgmod

        try:
            rm = cfgmod.resolve_cli_model(name, cfg)
        except cfgmod.ConfigError as e:
            raise SystemExit(f"error: {e}") from e
        if rm is not None:
            return rm.path
    raise SystemExit(f"error: {name}: no such GGUF file or configured model")


def _run_letters(model, processor, parsed, path: str) -> dict:
    """The letter readout on a text model, to completion on this thread."""
    import contextlib
    from typing import Any

    import mlx.core as mx

    from gmlx.systemone import jev_response, letters
    from gmlx.systemone.ar_reader import LetterReader, LetterTokens, decide_letters, run_to_end

    schema, state = parsed.get("letters")
    if letters.has_image(parsed.body.get("state")):
        raise SystemExit("error: images are not supported: the model is text-only")
    tokens = LetterTokens(processor)
    try:
        letter_ids = tokens.letter_ids()
    except ValueError as e:
        raise SystemExit(f"error: {path} cannot answer letter reads: {e}") from e
    wired: Any = contextlib.nullcontext()
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        wired = importlib.import_module("mlx_lm.generate").wired_limit(model)
    started = time.perf_counter()
    with wired:
        result = run_to_end(decide_letters(
            LetterReader(model, letter_ids), tokens, schema, state))
    print(f"[systemone] {result['diagnostics']['timing']['reads']} forwards in "
          f"{(time.perf_counter() - started) * 1e3:.0f} ms", file=sys.stderr)
    return jev_response(schema, result, 0, os.path.basename(path))


def _run_offline(a, body: dict) -> dict:
    from gmlx.systemone import Limits, TemplateResolver, decide, jev_response
    from gmlx.systemone.backends import ParsedRequest

    cfg, settings = _offline_settings(a.config)
    limits = Limits(max_questions=settings.max_questions,
                    max_samples=settings.max_samples)
    parsed = ParsedRequest(body, limits, settings.request_defaults())
    if parsed.error is not None:
        raise parsed.error
    path = _model_path(a.model, cfg)

    from gmlx.gen.diffusion import is_diffusion_model
    from gmlx.load.loader import load_model
    from gmlx.serve.bridge_vlm import _make_text_processor

    model, _config, tokenizer = load_model(path, verbose=False)
    processor = _make_text_processor(tokenizer)
    if not is_diffusion_model(model):
        return _run_letters(model, processor, parsed, path)
    schema, state, seed = parsed.get("diffusion")
    from gmlx.systemone.engine import (
        BoundReader,
        ChatTokens,
        StructuredReader,
        engine_scope,
    )

    gen = importlib.import_module("mlx_vlm.server.generation")
    canvas_len = min(int(settings.canvas), int(model.config.canvas_length))
    tokens = ChatTokens(processor, state)
    resolver = TemplateResolver(tokens.enc, canvas_len)
    started = time.perf_counter()
    with engine_scope(model, seed):
        reader = StructuredReader(
            model, prefill_step_size=int(gen.get_prefill_step_size()))
        engine = BoundReader(reader, processor=processor,
                             backend=processor.tokenizer)
        result, completion_tokens = decide(
            schema, state, engine=engine, resolver=resolver,
            chat_ids=tokens.chat_ids, seed=seed,
            constrained=settings.constrained, canvas_len=canvas_len,
            decode=tokens.decode)
    print(f"[systemone] {result['diagnostics']['timing']['reads']} reads in "
          f"{(time.perf_counter() - started) * 1e3:.0f} ms", file=sys.stderr)
    return jev_response(schema, result, completion_tokens, os.path.basename(path))


def cmd_systemone(argv: list | None = None, prog: str = "gmlx systemone") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Answer a structured-decision request (the POST "
                    "/v1/systemone body): on a running server by default, or "
                    "offline with --model.")
    ap.add_argument("request", metavar="REQUEST.json",
                    help="JSON file holding the request body.")
    ap.add_argument("--url", default=None, metavar="URL",
                    help="Server base URL (default: the single managed server "
                         "if there's one, else the config's host/port, else "
                         "http://127.0.0.1:8080).")
    ap.add_argument("--host", default=None, help="Server host (alternative to "
                    "--url).")
    ap.add_argument("--port", type=int, default=None, help="Server port "
                    "(alternative to --url).")
    ap.add_argument("--api-key", default=None, metavar="KEY",
                    help="API key for a key-protected server (default: the "
                         "GMLX_API_KEY env var).")
    ap.add_argument("--model", default=None, metavar="GGUF",
                    help="Run offline on this GGUF path or configured model id "
                         "instead of posting to a server. DiffusionGemma reads "
                         "answer slots, and any other text model answers with "
                         "the letter readout.")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Server config whose server.systemone settings and "
                         "model ids the offline run uses (default: the "
                         "standard config location).")
    ap.add_argument("--seed", type=int, default=None, metavar="N",
                    help="Seed for the reads; replaces the request's seed "
                         "(default: the request's seed, else 42).")
    ap.add_argument("--json", action="store_true",
                    help="Print the full response body as JSON.")
    a = ap.parse_args(argv)
    if a.model and (a.url or a.host or a.port is not None):
        ap.error("--model runs offline and cannot be combined with "
                 "--url, --host or --port")
    if a.config and not a.model:
        ap.error("--config applies only to an offline run with --model")

    try:
        with open(os.path.expanduser(a.request)) as f:
            body = json.load(f)
    except (OSError, ValueError) as e:
        print(f"error: {a.request}: {e}", file=sys.stderr)
        return 1
    if not isinstance(body, dict):
        print(f"error: {a.request}: expected a JSON object", file=sys.stderr)
        return 1
    if a.seed is not None:
        body["seed"] = a.seed

    if a.model:
        from gmlx.systemone import SchemaError

        try:
            out = _run_offline(a, body)
        except SchemaError as e:
            print(f"error: {a.request}: {e}", file=sys.stderr)
            return 1
    else:
        code, out = _post(a, body)
        if out is None:
            return code
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        _print_answers(out)
    return 0
