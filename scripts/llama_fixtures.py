#!/usr/bin/env python3
"""Regenerate the llama.cpp fixtures that gmlx's parity tests read.

Each subcommand runs one llama.cpp tool and writes one plain JSON fixture
under tests/fixtures/. The binaries come from environment variables:

  GMLX_LLAMA_SERVER        llama-server
  GMLX_LLAMA_PERPLEXITY    llama-perplexity

A fixture holds the llama.cpp commit (``llama_cpp``), the time it was made
(``created``), each run's argv keyed by a run label (``argv``), each file
it read by name, size and sha256 (``files``) and its rows. A row names its
run in ``run`` and its files by name only. Paths in an argv are reduced to
file names. Before it regenerates a fixture, the script checks every file
the old fixture names against its recorded sha256.

  GMLX_LLAMA_PERPLEXITY=.../llama-perplexity python scripts/llama_fixtures.py \\
      kld-reference --reference Qwen3.5-9B-BF16.gguf --model Qwen3.5-9B-Q4_K_M.gguf
  GMLX_LLAMA_SERVER=.../llama-server python scripts/llama_fixtures.py \\
      continuation Qwen3.5-9B-Q4_K_M.gguf gemma-4-12B-it-Q6_K.gguf \\
      SmolLM2-135M-Instruct-Q8_0.gguf
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

# Set, not defaulted: a fixture must not depend on the caller's environment.
# No import above loads mlx.
os.environ["MLX_ENABLE_TF32"] = "0"

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
BINARIES = {
    "llama-server": "GMLX_LLAMA_SERVER",
    "llama-perplexity": "GMLX_LLAMA_PERPLEXITY",
}
KLD_TEXT = FIXTURES / "kld" / "streaming.txt"
KLD_FIXTURE = FIXTURES / "kld" / "llama_perplexity.json"


def _image_fixtures():
    path = REPO / "scripts" / "image_oracle" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("_llama_image_fixtures", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def binary(tool: str) -> str:
    var = BINARIES[tool]
    path = os.environ.get(var, "")
    if not path:
        raise SystemExit(f"error: set {var} to the {tool} binary")
    if not os.access(path, os.X_OK):
        raise SystemExit(f"error: {var}={path!r} is not an executable file")
    return path


def llama_commit(path: str) -> str:
    """The commit a llama.cpp binary prints for --version."""
    out = subprocess.run([path, "--version"], capture_output=True, text=True,
                         check=False)
    m = re.search(r"commit ([0-9a-f]+)", out.stdout + out.stderr)
    if not m:
        raise SystemExit(f"error: no commit in {path} --version output")
    return m.group(1)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_recorded_files(fixture: Path, paths: list[Path]) -> None:
    """Refuse when a file the old fixture names has another sha256 now."""
    if not fixture.is_file():
        return
    old = json.loads(fixture.read_text()).get("files", {})
    ident = _image_fixtures().file_identity
    for p in paths:
        rec = old.get(p.name)
        if rec is None:
            continue
        cur = ident(p)
        if cur["sha256"] != rec.get("sha256"):
            raise SystemExit(
                f"error: {p.name} differs from the file {fixture.name} "
                f"recorded (sha256 {cur['sha256'][:12]} vs "
                f"{str(rec.get('sha256'))[:12]}); remove the fixture to "
                "record the new file")


def _shown(path: Path) -> Path:
    path = path.resolve()
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def write_fixture(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=1,
                               ensure_ascii=True) + "\n")
    print(f"wrote {_shown(path)}")


def run_logged(argv: list[str], log: Path) -> str:
    print("$ " + " ".join(argv), flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                            check=False).returncode
    text = log.read_text(errors="replace")
    if rc != 0:
        raise SystemExit(f"error: {Path(argv[0]).name} exited {rc}; see {log}")
    return text


def read_base_tokens(path: Path) -> tuple[int, int, int, list[int]]:
    """(n_ctx, n_vocab, n_chunk, tokens) from a --kl-divergence-base file."""
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            raise SystemExit(f"error: {path} is not a llama.cpp logits file")
        (n_ctx,) = struct.unpack("<I", f.read(4))
        n_vocab, n_chunk = struct.unpack("<ii", f.read(8))
        raw = f.read(4 * n_ctx * n_chunk)
    tokens = list(struct.unpack(f"<{n_ctx * n_chunk}i", raw))
    return n_ctx, n_vocab, n_chunk, tokens


def tokens_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}i", *tokens)).hexdigest()


_KLD_RE = re.compile(r"Mean\s+KLD:\s*([-0-9.eE+]+)\s*\S+\s*([-0-9.eE+]+)")
_TOP_RE = re.compile(r"Same top p:\s*([-0-9.eE+]+)\s*\S+\s*([-0-9.eE+]+)\s*%")


def parse_kld(text: str) -> dict:
    """Mean KLD and Same top p from a --kl-divergence run, the share as a
    fraction from 0 to 1."""
    k = _KLD_RE.search(text)
    t = _TOP_RE.search(text)
    if not (k and t):
        raise SystemExit("error: no Mean KLD or Same top p in the output")
    return {"mean_kld": float(k.group(1)), "mean_kld_unc": float(k.group(2)),
            "same_top_p": float(t.group(1)) / 100.0,
            "same_top_p_unc": float(t.group(2)) / 100.0}


def cmd_kld_reference(args) -> int:
    tool = binary("llama-perplexity")
    ref = Path(args.reference).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    text = Path(args.text).expanduser().resolve()
    out = Path(args.out)
    for p in (ref, model, text):
        if not p.is_file():
            raise SystemExit(f"error: no such file: {p}")
    check_recorded_files(out, [ref, model, text])
    commit = llama_commit(tool)
    ident = _image_fixtures().file_identity
    portable = _image_fixtures().portable_argv
    chunks = ["--chunks", str(args.chunks)] if args.chunks else []
    with tempfile.TemporaryDirectory(prefix="llama-kld-",
                                     dir=args.work_dir) as tmp:
        base = Path(tmp) / "reference.kld"
        argv_base = [tool, "-m", str(ref), "-f", str(text), "-c", str(args.ctx),
                     *chunks, "--kl-divergence-base", str(base)]
        log_base = Path(args.log_dir) / "kld-reference-base.log"
        run_logged(argv_base, log_base)
        n_ctx, n_vocab, n_chunk, tokens = read_base_tokens(base)
        argv_kld = [tool, "-m", str(model), "-c", str(args.ctx),
                    "--kl-divergence-base", str(base), "--kl-divergence"]
        log_kld = Path(args.log_dir) / "kld-reference-kld.log"
        stats = parse_kld(run_logged(argv_kld, log_kld))
    data = {
        "llama_cpp": commit,
        "created": now(),
        "argv": {"llama-perplexity/base": portable(argv_base),
                 "llama-perplexity/kld": portable(argv_kld)},
        "files": {p.name: ident(p) for p in (ref, model, text)},
        "rows": [{
            "run": "llama-perplexity/kld",
            "base_run": "llama-perplexity/base",
            "model": model.name,
            "reference": ref.name,
            "text": text.name,
            "ctx": n_ctx,
            "chunks": n_chunk,
            "n_vocab": n_vocab,
            "tokens_sha256": tokens_sha256(tokens),
            "tokens_head": tokens[:32],
            **stats,
        }],
    }
    write_fixture(out, data)
    return 0


def cmd_kld_adapter(args) -> int:
    """llama-perplexity's Mean KLD and Same top p of MODEL with a rank-1
    adapter at each strength, against MODEL bare, over the frozen docs
    page. The adapter is minted once, on ffn_down of a few layers, and
    checked in; a later run reuses it. The fixture records the layers and
    the B norm read back from the adapter, and the seed only when this run
    minted it."""
    import gguf
    import numpy as np

    layers, norm, seed = (8, 16, 24), 60.0, 0
    tool = binary("llama-perplexity")
    model = Path(args.model).expanduser().resolve()
    text = Path(args.text).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    out = Path(args.out)
    for p in (model, text):
        if not p.is_file():
            raise SystemExit(f"error: no such file: {p}")
    minted = not adapter.is_file()
    if minted:
        # A unit A and a B of norm `norm` per layer: delta = B A, alpha 1.
        reader = gguf.GGUFReader(str(model))
        arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
        shapes = {t.name: [int(n) for n in t.shape] for t in reader.tensors}
        rng = np.random.default_rng(seed)
        w = gguf.GGUFWriter(str(adapter), arch)
        w.add_type(gguf.GGUFType.ADAPTER)
        w.add_string(gguf.Keys.Adapter.TYPE, "lora")
        w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 1.0)
        for i in layers:
            n_in, n_out = shapes[f"blk.{i}.ffn_down.weight"]
            a = rng.standard_normal((1, n_in))
            b = rng.standard_normal((n_out, 1))
            w.add_tensor(f"blk.{i}.ffn_down.weight.lora_a",
                         (a / np.linalg.norm(a)).astype(np.float32))
            w.add_tensor(f"blk.{i}.ffn_down.weight.lora_b",
                         (b * norm / np.linalg.norm(b)).astype(np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"wrote {_shown(adapter)}")
    found: dict = {}
    for t in gguf.GGUFReader(str(adapter)).tensors:
        m = re.fullmatch(r"blk\.(\d+)\..*\.lora_b", t.name)
        if m:
            found[int(m[1])] = round(float(np.linalg.norm(np.asarray(t.data))), 4)
    provenance = {"adapter_layers": sorted(found),
                  "adapter_b_norm": sorted(set(found.values()))}
    if len(provenance["adapter_b_norm"]) == 1:
        provenance["adapter_b_norm"] = provenance["adapter_b_norm"][0]
    if minted:
        provenance["adapter_seed"] = seed
    check_recorded_files(out, [model, adapter, text])
    commit = llama_commit(tool)
    ident = _image_fixtures().file_identity
    portable = _image_fixtures().portable_argv
    chunks = ["--chunks", str(args.chunks)] if args.chunks else []
    argvs, rows = {}, []
    with tempfile.TemporaryDirectory(prefix="llama-kld-adapter-",
                                     dir=args.work_dir) as tmp:
        base = Path(tmp) / "reference.kld"
        argv = [tool, "-m", str(model), "-f", str(text), "-c", str(args.ctx),
                *chunks, "--kl-divergence-base", str(base)]
        run_logged(argv, Path(args.log_dir) / "kld-adapter-base.log")
        argvs["llama-perplexity/base"] = portable(argv)
        n_ctx, n_vocab, n_chunk, tokens = read_base_tokens(base)
        for strength in args.strength:
            run = f"llama-perplexity/s{strength:g}"
            argv = [tool, "-m", str(model), "-c", str(args.ctx),
                    "--kl-divergence-base", str(base), "--kl-divergence",
                    "--lora-scaled", f"{adapter}:{strength:g}"]
            log = Path(args.log_dir) / f"kld-adapter-s{strength:g}.log"
            stats = parse_kld(run_logged(argv, log))
            argvs[run] = portable(argv)
            rows.append({"run": run, "base_run": "llama-perplexity/base",
                         "model": model.name, "adapter": adapter.name,
                         "strength": strength, "text": text.name,
                         "ctx": n_ctx, "chunks": n_chunk, "n_vocab": n_vocab,
                         "tokens_sha256": tokens_sha256(tokens), **stats})
    write_fixture(out, {
        "llama_cpp": commit, "created": now(), "argv": argvs,
        "files": {p.name: ident(p) for p in (model, adapter, text)},
        **provenance, "rows": rows})
    return 0


# continuation: llama-server /apply-template on a final assistant message.
CONTINUATION_FIXTURE = FIXTURES / "llama_continuation.json"
# GGUF file name -> (run label suffix, template fixture, thinking values).
# None sends no enable_thinking: the template has no switch.
CONTINUATION_GGUFS = {
    "Qwen3.5-9B-Q4_K_M.gguf": ("qwen35_9b", "qwen35_9b_template.jinja",
                               (False, True)),
    "gemma-4-12B-it-Q6_K.gguf": ("gemma4_12b", "gemma4_12b_template.jinja",
                                 (False, True)),
    "SmolLM2-135M-Instruct-Q8_0.gguf": ("smollm2", "smollm2_template.jinja",
                                        (None,)),
}
CONTINUATION_HISTORY = [{"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "Hi"}]
_C = {"role": "assistant", "content": "Hello"}
_R = {"role": "assistant", "content": "", "reasoning_content": "plan"}
_RC = {"role": "assistant", "content": "Hello", "reasoning_content": "plan"}
# Case name -> (final message or None, continue_final_message or None).
CONTINUATION_CASES = {
    "content": (_C, True),
    "reasoning": (_R, True),
    "reasoning-and-content": (_RC, True),
    "string-content": (_RC, "content"),
    "string-reasoning": (_RC, "reasoning_content"),
    "no-generation-prompt": (None, None),
}
# llama.cpp's log name of each dedicated handler -> gmlx's handler name
# (gmlx.gen.continuation.handler_for).
_HANDLER_LOG_NAMES = {
    "Ministral/Magistral Large 3": "ministral-3",
    "LLM-jp Harmony v1": "llm-jp-harmony",
    "Harmony XML": "harmony-xml",
    "GPT-OSS": "gpt-oss",
    "Muse Spark": "muse-spark",
    "Muse Glimmer": "muse-glimmer",
    "Functionary v3.2": "functionary-v3-2",
    "Kimi K2 Thinking": "kimi-k2",
    "Kimi K3": "kimi-k3",
    "K2 Horizon": "k2-horizon",
    "Ling 3.0 (Bailing V3)": "ling3",
    "Cohere2 MoE": "cohere2moe",
    "LFM2": "lfm2",
    "LFM2.5": "lfm2-5",
    "GigaChatV3": "gigachat-v3",
    "MiniMax-M3": "minimax-m3",
    "DeepSeek V3.2/V4": "deepseek-v3-2",
    "Gemma4": "gemma4",
    "MiniCPM5": "minicpm5",
    "TranslateGemma": "translate-gemma",
    "Qwen3-Coder": "qwen3-coder",
}
_HANDLER_RE = re.compile(
    r"Using specialized template: (.+?)\s*$|using differential autoparser",
    re.M)


def gguf_template(path: Path) -> tuple[str, dict]:
    """The chat template of a GGUF and what decides its BOS cut."""
    sys.path.insert(0, str(REPO))
    from gmlx.load.headerscan import scan_gguf

    kv = scan_gguf(str(path), include_tensors=False, array_limit=1 << 20).kv
    tmpl = kv.get("tokenizer.chat_template")
    if not isinstance(tmpl, str) or not tmpl:
        raise SystemExit(f"error: {path.name} has no tokenizer.chat_template")
    tokens = kv.get("tokenizer.ggml.tokens")
    bos_id = kv.get("tokenizer.ggml.bos_token_id")
    bos = tokens[bos_id] if tokens is not None and bos_id is not None else ""
    return tmpl, {"add_bos": bool(kv.get("tokenizer.ggml.add_bos_token")),
                  "bos": bos}


def check_template_fixture(name: str, text: str) -> Path:
    """Write the template as a new fixture, or refuse when an existing one
    differs."""
    path = FIXTURES / name
    if path.is_file():
        if path.read_text() != text:
            raise SystemExit(
                f"error: {name} differs from the template in the GGUF; remove "
                "it to record the GGUF's template")
    else:
        path.write_text(text)
        print(f"wrote {_shown(path)}")
    return path


def continuation_body(case: str, thinking) -> dict:
    final, cont = CONTINUATION_CASES[case]
    body: dict = {"messages": [*CONTINUATION_HISTORY]
                  + ([final] if final else []),
                  "add_generation_prompt": False}
    if cont is not None:
        body["continue_final_message"] = cont
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    return body


class LlamaServer:
    """One llama-server process, its output in a log file. Ready once it
    logs the address it listens on."""

    def __init__(self, argv: list[str], log: Path):
        self.log_path = log
        self.lines: list[str] = []
        self._ready = threading.Event()
        self._lock = threading.Lock()
        print("$ " + " ".join(argv), flush=True)
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     errors="replace")
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        with open(self.log_path, "w") as f:
            for line in self.proc.stdout:
                f.write(line)
                f.flush()
                with self._lock:
                    self.lines.append(line)
                if "listening on" in line:
                    self._ready.set()
        self._ready.set()

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout) or self.proc.poll() is not None:
            self.stop()
            raise SystemExit(f"error: llama-server did not start; see "
                             f"{self.log_path}")

    def output(self) -> str:
        with self._lock:
            return "".join(self.lines)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._reader.join(timeout=60)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_json(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def logged_handler(startup: str, template: str) -> str:
    """The handler class llama-server logged as it rendered the template's
    example at startup. A dedicated handler logs once, when the template
    loads, and the autoparser on each render. The gemma4 handler logs the
    legacy case at another level, so the template source decides it, as in
    common/chat.cpp."""
    found = list(_HANDLER_RE.finditer(startup))
    if not found:
        raise SystemExit("error: llama-server logged no chat handler; it "
                         "needs -v")
    name = found[-1].group(1)
    if name is None:
        return "autoparser"
    if name not in _HANDLER_LOG_NAMES:
        raise SystemExit(f"error: unknown llama.cpp handler {name!r}")
    handler = _HANDLER_LOG_NAMES[name]
    if handler == "gemma4" and "{#- OpenAI Chat Completions:" not in template:
        return "gemma4-legacy"
    return handler


def cmd_continuation(args) -> int:
    tool = binary("llama-server")
    ggufs = [Path(p).expanduser().resolve() for p in args.gguf]
    for p in ggufs:
        if not p.is_file():
            raise SystemExit(f"error: no such file: {p}")
        if p.name not in CONTINUATION_GGUFS:
            raise SystemExit(
                f"error: {p.name} is not a continuation GGUF; the script "
                f"knows {', '.join(sorted(CONTINUATION_GGUFS))}")
    out = Path(args.out)
    templates = [FIXTURES / CONTINUATION_GGUFS[p.name][1] for p in ggufs]
    check_recorded_files(out, ggufs + [t for t in templates if t.is_file()])
    commit = llama_commit(tool)
    ident = _image_fixtures().file_identity
    portable = _image_fixtures().portable_argv
    argvs, files, rows = {}, {}, []
    for gguf in ggufs:
        label, tmpl_name, thinking_values = CONTINUATION_GGUFS[gguf.name]
        text, bos = gguf_template(gguf)
        tmpl = check_template_fixture(tmpl_name, text)
        run = f"llama-server/{label}"
        port = str(free_port())
        # -m loads the same GGUF, so bos_token and add_bos come from it.
        argv = [tool, "-m", str(gguf), "--jinja", "--chat-template-file",
                str(tmpl), "--host", "127.0.0.1", "--port", port,
                "-c", "512", "-ngl", "0", "--no-warmup", "-v"]
        server = LlamaServer(
            argv, Path(args.log_dir) / f"continuation-{label}.log")
        try:
            server.wait_ready(args.start_timeout)
            handler = logged_handler(server.output(), text)
            for thinking in thinking_values:
                tag = {None: "", False: "/think-off", True: "/think-on"}[thinking]
                for case in CONTINUATION_CASES:
                    body = continuation_body(case, thinking)
                    status, reply = post_json(
                        f"http://127.0.0.1:{port}/apply-template", body)
                    if status != 200:
                        raise SystemExit(
                            f"error: {label} {case}: HTTP {status}: {reply}")
                    prompt = reply["prompt"]
                    rows.append({
                        "id": f"{label}{tag}/{case}",
                        "run": run,
                        "gguf": gguf.name,
                        "template": tmpl.name,
                        "handler": handler,
                        "body": body,
                        "prompt": prompt,
                        # llama.cpp cuts a leading BOS text from the render
                        # when the vocab adds BOS (common/chat.cpp).
                        "bos_cut": bool(bos["add_bos"] and bos["bos"]
                                        and "bos_token" in text
                                        and not prompt.startswith(bos["bos"])),
                    })
        finally:
            server.stop()
        argvs[run] = portable([a if a != port else "PORT" for a in argv])
        files[gguf.name] = ident(gguf)
        files[tmpl.name] = ident(tmpl)
    write_fixture(out, {"llama_cpp": commit, "created": now(), "argv": argvs,
                        "files": files, "rows": rows})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    kr = sub.add_parser(
        "kld-reference",
        help="llama-perplexity KL divergence of a model against a reference "
        "file over a frozen docs page")
    kr.add_argument("--reference", required=True,
                    help="the reference GGUF (--kl-divergence-base)")
    kr.add_argument("--model", required=True,
                    help="the GGUF scored against it (--kl-divergence)")
    kr.add_argument("--text", default=str(KLD_TEXT))
    kr.add_argument("--ctx", type=int, default=512)
    kr.add_argument("--chunks", type=int, default=None)
    kr.add_argument("--out", default=str(KLD_FIXTURE))
    kr.add_argument("--work-dir", default=None,
                    help="where the reference logits file goes (default: the "
                    "temporary directory)")
    kr.add_argument("--log-dir", default=".",
                    help="where each run's output log goes")
    kr.set_defaults(fn=cmd_kld_reference)
    ka = sub.add_parser(
        "kld-adapter",
        help="llama-perplexity KL divergence of a model with a checked-in "
        "rank-1 adapter at each strength, against the model bare")
    ka.add_argument("--model", required=True,
                    help="the GGUF the adapter is minted for and scored on")
    ka.add_argument("--adapter",
                    default=str(FIXTURES / "kld" / "qwen35_9b_rank1.gguf"),
                    help="the adapter; minted here when it does not exist")
    ka.add_argument("--strength", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0])
    ka.add_argument("--text", default=str(KLD_TEXT))
    ka.add_argument("--ctx", type=int, default=512)
    ka.add_argument("--chunks", type=int, default=None)
    ka.add_argument("--out", default=str(FIXTURES / "kld"
                                         / "llama_perplexity_adapter.json"))
    ka.add_argument("--work-dir", default=None,
                    help="where the reference logits file goes (default: the "
                    "temporary directory)")
    ka.add_argument("--log-dir", default=".",
                    help="where each run's output log goes")
    ka.set_defaults(fn=cmd_kld_adapter)
    co = sub.add_parser(
        "continuation",
        help="llama-server /apply-template of a final assistant message "
        "continued, for each template, thinking on and off and each mode")
    co.add_argument("gguf", nargs="+",
                    help="the GGUFs whose templates the rows render")
    co.add_argument("--out", default=str(CONTINUATION_FIXTURE))
    co.add_argument("--start-timeout", type=float, default=600.0,
                    help="seconds to wait for llama-server to load a GGUF")
    co.add_argument("--log-dir", default=".",
                    help="where each server's output log goes")
    co.set_defaults(fn=cmd_continuation)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
