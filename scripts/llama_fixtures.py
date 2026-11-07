#!/usr/bin/env python3
"""Regenerate the llama.cpp fixtures that gmlx's parity tests read.

Each subcommand runs one llama.cpp tool and writes one plain JSON fixture
under tests/fixtures/. The binaries come from environment variables:

  GMLX_LLAMA_SERVER        llama-server
  GMLX_LLAMA_EXPORT_LORA   llama-export-lora
  GMLX_LLAMA_PERPLEXITY    llama-perplexity

A fixture holds the llama.cpp commit (``llama_cpp``), the time it was made
(``created``), each run's argv keyed by a run label (``argv``), each file
it read by name, size and sha256 (``files``) and its rows. A row names its
run in ``run`` and its files by name only. Paths in an argv are reduced to
file names. Before it regenerates a fixture, the script checks every file
the old fixture names against its recorded sha256.

  GMLX_LLAMA_PERPLEXITY=.../llama-perplexity python scripts/llama_fixtures.py \\
      kld-reference --reference Qwen3.5-9B-BF16.gguf --model Qwen3.5-9B-Q4_K_M.gguf
"""

from __future__ import annotations

import os

# gmlx runs inside a subcommand compute in exact float32.
os.environ["MLX_ENABLE_TF32"] = "0"

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
BINARIES = {
    "llama-server": "GMLX_LLAMA_SERVER",
    "llama-export-lora": "GMLX_LLAMA_EXPORT_LORA",
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


def write_fixture(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=1,
                               ensure_ascii=True) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


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
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
