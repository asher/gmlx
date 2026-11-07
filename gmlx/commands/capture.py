"""``gmlx capture``: the residual stream after each block, over prompt files.

Each prompt renders as an optional system message and one user message,
with a generation prompt, through the serve path's own render
(``prompt_utils.apply_chat_template``) and tokenize step
(``ResponseGenerator._cpu_preprocess``). Each forward runs one prompt
with a fresh cache. The output is safetensors: ``<set>.mean``
``[n_layer, n_embd]`` f32 for every set, and ``<set>.rows``
``[n_layer, n_prompts, n_embd]`` f16 for the sets named in ``--rows``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import sys
import types

_SET_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _build_parser(prog: str) -> argparse.ArgumentParser:
    from .cli import _DtypeAction, add_rope_args

    ap = argparse.ArgumentParser(
        prog=prog,
        description="Record the residual stream after each decoder block, "
                    "averaged over the last prompt tokens, for named sets of "
                    "prompts, into a safetensors file.")
    ap.add_argument("gguf", metavar="MODEL",
                    help="GGUF path, or a model id or alias from the config.")
    ap.add_argument("--prompts", action="append", required=True,
                    metavar="NAME=FILE",
                    help="A named prompt set: a JSONL file of one JSON string "
                         "per line. Repeat for more sets; all run in one "
                         "model load.")
    ap.add_argument("--out", required=True, metavar="FILE",
                    help="Write the safetensors file here.")
    ap.add_argument("--positions", type=int, default=1, metavar="N",
                    help="Average the block outputs over the last N prompt "
                         "tokens (default 1).")
    ap.add_argument("--rows", action="append", default=[], metavar="SET",
                    help="Also write the per-prompt rows of this set, as "
                         "<set>.rows. Repeatable, or a comma list.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Use only the first N prompts of each set.")
    ap.add_argument("--system", default=None, metavar="TEXT",
                    help="Render each prompt after this system message.")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Render with thinking on (--think) or off "
                         "(--no-think), set through the variable this "
                         "model's template reads (default: the template's "
                         "default).")
    ap.add_argument("--chat-template", default=None, metavar="STR|PATH",
                    help="Inline Jinja chat template, or a path to a "
                         ".jinja/.txt file, replacing the GGUF's template.")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Server config to resolve a model id against when "
                         "the positional is not a file (default: the first "
                         "existing default config).")
    ap.add_argument("--json", action="store_true",
                    help="Print one JSON line per prompt and a last result "
                         "line on stdout, each with an \"event\" key of "
                         "\"progress\" or \"result\".")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Full load diagnostics instead of the progress "
                         "spinner.")
    grp = ap.add_argument_group("model placement")
    placement = grp.add_mutually_exclusive_group()
    placement.add_argument("--stream-experts", action="store_true",
                           help="Stream a MoE model's routed-expert stacks "
                                "from disk while the every-token layers stay "
                                "on the GPU, as in serve.")
    placement.add_argument("--stream-cpu", action="store_true",
                           help="Run the whole model on the CPU device with "
                                "every weight streamed from the page cache, "
                                "as in serve.")
    grp.add_argument("--prefill-feeder", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="Stage each layer's expert stacks straight from "
                          "the GGUF for a streamed model (default on).")
    grp.add_argument("--dtype", choices=("auto", "bfloat16", "float16"),
                     default=None, action=_DtypeAction, metavar="DTYPE",
                     help="Activation dtype (default auto: float16 on M1 "
                          "and M2, bfloat16 elsewhere).")
    add_rope_args(ap)
    # A config entry's adapter lands here, and capture refuses it.
    ap.set_defaults(adapter=None)
    return ap


def _parse_sets(ap, specs: list[str]) -> list[tuple[str, str]]:
    sets: list[tuple[str, str]] = []
    for spec in specs:
        name, eq, path = spec.partition("=")
        if not eq or not path:
            ap.error(f"--prompts takes NAME=FILE, got {spec!r}")
        if not _SET_NAME.fullmatch(name):
            ap.error(f"--prompts: set name {name!r} may hold only letters, "
                     "digits, '_' and '-'")
        if any(name == n for n, _ in sets):
            ap.error(f"--prompts: set {name!r} is named twice")
        sets.append((name, os.path.expanduser(path)))
    return sets


def read_prompts(path: str, limit: int | None = None) -> list[str]:
    """The prompts of a JSONL file: one JSON string per line, blank lines
    skipped. Raises ValueError naming the line of a bad entry."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: not JSON: {e}") from None
            if not isinstance(value, str):
                raise ValueError(f"{path}:{n}: expected a JSON string, got "
                                 f"{type(value).__name__}")
            out.append(value)
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"{path}: holds no prompts")
    return out


def template_kwargs(processor, think) -> dict:
    """The template variables for ``think`` (True, False or None), spelled
    as this model's template reads them."""
    from gmlx.serve.patches.chat_behavior import _model_template_text
    from gmlx.tui.reasoning import map_thinking_controls

    mode = None if think is None else ("on" if think else "off")
    return map_thinking_controls(
        {}, mode, None, _model_template_text(processor),
        warn=lambda msg: print(f"[thinking] {msg}", file=sys.stderr))


def prompt_ids(model, processor, config, user: str, system: str | None,
               kwargs: dict) -> list[int]:
    """One prompt's token ids, rendered and tokenized as serve does for a
    chat request with these messages."""
    from gmlx._vendor.mlx_vlm import prompt_utils

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    messages = ([{"role": "system", "content": system}] if system is not None else [])
    messages.append({"role": "user", "content": user})
    text = prompt_utils.apply_chat_template(
        processor, config, messages, add_generation_prompt=True,
        num_images=0, num_audios=0, **kwargs)
    inputs = gen.ResponseGenerator._cpu_preprocess(
        types.SimpleNamespace(processor=processor, model=model), text)
    return [int(t) for t in inputs["input_ids"].reshape(-1).tolist()]


def install_render_fixes() -> None:
    """The serve installs that change how a plain chat prompt renders and
    encodes: the template's own thinking default, a system message as a
    string, and one BOS."""
    from gmlx.tui.reasoning import install_template_default_thinking
    from gmlx.upstream.single_bos_fix import install_single_bos_fix
    from gmlx.upstream.system_content_fix import install_system_content_fix

    install_template_default_thinking()
    install_system_content_fix()
    install_single_bos_fix()


def gguf_facts(path: str) -> dict:
    """``arch`` (the GGUF's general.architecture), ``n_layer`` (blocks
    without the nextn ones, or None), ``file`` and ``file_size`` (every
    shard)."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    kv = scan_gguf(path, include_tensors=False).kv
    arch = kv.get("general.architecture")
    blocks = kv.get(f"{arch}.block_count")
    nextn = kv.get(f"{arch}.nextn_predict_layers") or 0
    n_layer = int(blocks) - int(nextn) if isinstance(blocks, int) else None
    size = sum(os.path.getsize(p) for p in find_split_shards(path))
    return {"arch": arch, "n_layer": n_layer, "file": os.path.basename(path),
            "file_size": size}


def load(args):
    """``(model, processor, config)`` through serve's text load, with the
    placement the flags ask for."""
    import gmlx.load.loadlog as loadlog
    from gmlx.serve.bridge_vlm import _model_type, load_serveable_model
    from gmlx.upstream.system_content_fix import enable_system_content_fix

    from .cli import rope_override_from_args

    stream = "cpu" if args.stream_cpu else ("experts" if args.stream_experts else None)
    with loadlog.load_ui(args.verbose, args.gguf):
        model, processor, config = load_serveable_model(
            args.gguf, chat_template=args.chat_template, stream=stream,
            feeder_prefill=args.prefill_feeder,
            rope_override=rope_override_from_args(args))
    enable_system_content_fix(processor, _model_type(config))
    return model, processor, config


class _Emitter:
    """Progress and result output. With --json each event is one JSON line
    on the real stdout, and everything else printed goes to stderr."""

    def __init__(self, as_json: bool):
        self.as_json = as_json
        self.out = sys.stdout
        self.tty = sys.stderr.isatty()

    def quiet_stdout(self):
        if self.as_json:
            return contextlib.redirect_stdout(sys.stderr)
        return contextlib.nullcontext()

    def progress(self, name: str, done: int, total: int, tokens: int) -> None:
        if self.as_json:
            self._line({"event": "progress", "set": name, "done": done,
                        "total": total, "tokens": tokens})
        elif self.tty:
            end = "\n" if done == total else ""
            print(f"\r[capture] {name}: {done}/{total}", end=end,
                  file=sys.stderr, flush=True)

    def result(self, body: dict) -> None:
        if self.as_json:
            self._line({"event": "result", **body})
            return
        sets = ", ".join(f"{k} ({v})" for k, v in body["sets"].items())
        print(f"[capture] wrote {body['out']}: {body['n_layer']} layers x "
              f"{body['n_embd']}, sets {sets}", file=self.out)

    def _line(self, obj: dict) -> None:
        self.out.write(json.dumps(obj) + "\n")
        self.out.flush()


def run_capture(model, processor, config, sets, *, positions: int,
                rows: set, system: str | None, think, n_layer: int | None,
                emit: _Emitter):
    """Capture every set. Returns ``(arrays, sizes, n_embd, n_layer)``."""
    import mlx.core as mx

    from gmlx.serve import cb_phase
    from gmlx.tune.capture import block_outputs, capture_layers, layer_capture

    layers = capture_layers(model, n_layer)
    if not layers:
        raise RuntimeError("found no decoder layers to capture in this model")
    kwargs = template_kwargs(processor, think)
    arrays: dict = {}
    sizes: dict = {}
    # As serve's engine jobs run a forward: the prefill caps, and the wired
    # limit for a model on the GPU (a streamed load leaves it at 0).
    wired = contextlib.nullcontext()
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        wired = importlib.import_module("gmlx._vendor.mlx_lm.generate").wired_limit(model)
    before = cb_phase.snapshot()
    cb_phase.flip("prefill", model)
    try:
        with wired, layer_capture(layers, positions) as rec:
            for name, prompts in sets:
                total = None
                per_prompt = []
                for i, text in enumerate(prompts):
                    ids = prompt_ids(model, processor, config, text, system, kwargs)
                    out = block_outputs(model, ids, rec)
                    total = out if total is None else total + out
                    if name in rows:
                        per_prompt.append(out.astype(mx.float16))
                    mx.eval(total, *per_prompt[-1:])
                    emit.progress(name, i + 1, len(prompts), len(ids))
                arrays[f"{name}.mean"] = total / len(prompts)
                if name in rows:
                    arrays[f"{name}.rows"] = mx.stack(per_prompt, axis=1)
                sizes[name] = len(prompts)
                mx.eval(*arrays.values())
    finally:
        cb_phase.restore(before)
    return arrays, sizes, int(total.shape[1]), len(layers)


def save(path: str, arrays: dict, metadata: dict) -> None:
    """Write ``arrays`` as safetensors at ``path``, through a temporary file
    in the same folder (mlx adds the suffix to a name without one)."""
    import mlx.core as mx

    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                       f".{os.path.basename(path)}.{os.getpid()}.safetensors")
    try:
        mx.save_safetensors(tmp, arrays, metadata=metadata)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cmd_capture(argv: list | None = None, prog: str = "gmlx capture") -> int:
    ap = _build_parser(prog)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    sets_spec = _parse_sets(ap, args.prompts)
    names = [n for n, _ in sets_spec]
    rows = {r.strip() for spec in args.rows for r in spec.split(",") if r.strip()}
    unknown = sorted(rows - set(names))
    if unknown:
        ap.error(f"--rows names no --prompts set: {', '.join(unknown)}")
    if args.positions < 1:
        ap.error("--positions must be at least 1")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    out_dir = os.path.dirname(os.path.abspath(os.path.expanduser(args.out)))
    if not os.path.isdir(out_dir):
        ap.error(f"--out: no folder {out_dir}")

    emit = _Emitter(args.json)
    with emit.quiet_stdout():
        from .cli import _ensure_stream_cb_caps, maybe_load_from_config

        rc = maybe_load_from_config(args, ap, argv)
        if rc is not None:
            return rc
        if args.adapter:
            print(f"error: the config entry sets an adapter ({args.adapter}), "
                  "which gmlx capture does not apply", file=sys.stderr)
            return 2
        path = os.path.expanduser(args.gguf)
        if not os.path.isfile(path):
            print(f"error: no GGUF file or configured model {args.gguf!r}",
                  file=sys.stderr)
            return 2
        args.gguf = path
        try:
            sets = [(name, read_prompts(p, args.limit)) for name, p in sets_spec]
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        facts = gguf_facts(path)
        _ensure_stream_cb_caps(args)
        install_render_fixes()
        model, processor, config = load(args)
        arrays, sizes, n_embd, n_layer = run_capture(
            model, processor, config, sets, positions=args.positions,
            rows=rows, system=args.system, think=args.think,
            n_layer=facts["n_layer"], emit=emit)
        import gmlx

        metadata = {
            "format": "gmlx.capture",
            "arch": str(facts["arch"]),
            "file": facts["file"],
            "file_size": str(facts["file_size"]),
            "sets": json.dumps(sizes),
            "positions": str(args.positions),
            "think": {None: "default", True: "on", False: "off"}[args.think],
            "gmlx_version": gmlx.__version__,
        }
        out = os.path.expanduser(args.out)
        save(out, arrays, metadata)
    emit.result({"out": out, "n_layer": n_layer, "n_embd": n_embd,
                 "sets": sizes, "rows": sorted(rows)})
    return 0
