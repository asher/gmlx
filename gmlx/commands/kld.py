"""``gmlx kld``: the KL divergence over a text from a reference model to
MODEL, as llama.cpp's ``llama-perplexity --kl-divergence`` computes it.

The reference loads first, alone, and scores the text into a temporary
file of uint16 log-prob records. It unloads, then MODEL loads and scores
against that file. ``gmlx.gen.kld`` holds the arithmetic.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from gmlx.gen import kld as K

# The float32 logits of one head pass stay under this budget.
ROW_BUDGET_BYTES = 1 << 30
# Vocabulary rows per float32 head block.
HEAD_BLOCK = 16384
# Free disk the reference file leaves on its volume.
DISK_MARGIN_BYTES = 1 << 30


class Refusal(Exception):
    """A run the command refuses, with the reason."""


def build_parser(prog: str = "gmlx kld") -> argparse.ArgumentParser:
    from .cli import add_placement_args, add_verbosity_arg

    ap = argparse.ArgumentParser(
        prog=prog,
        description="Report the KL divergence over a text from a reference "
        "GGUF to MODEL, as llama-perplexity --kl-divergence computes it: "
        "Mean KLD and Same top p.",
    )
    ap.add_argument("model", metavar="MODEL",
                    help="The GGUF to score, or a model id or alias from "
                    "your server config.")
    ap.add_argument("--reference", metavar="FILE", default=None,
                    help="The GGUF to score MODEL against, such as the BF16 "
                    "file of the same model. Required.")
    ap.add_argument("--text", metavar="FILE", required=True,
                    help="The UTF-8 text to score. It must hold at least "
                    "2 x --ctx tokens.")
    ap.add_argument("--ctx", type=int, default=512, metavar="N",
                    help="Tokens per chunk (default: 512). Each chunk scores "
                    "its positions from N/2 to N-2.")
    ap.add_argument("--chunks", type=int, default=None, metavar="N",
                    help="Score at most N chunks (default: every whole chunk "
                    "of the text).")
    ap.add_argument("--json", action="store_true",
                    help="Print JSON lines: one progress line per chunk and "
                    "one result line, each with an \"event\" key.")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Server config to resolve a model name against when "
                    "MODEL or --reference is not a file (default: the first "
                    "existing default config).")
    add_placement_args(ap)
    add_verbosity_arg(ap)
    return ap


@dataclass
class Source:
    """A model to load: its GGUF and the server-config entry it came from,
    if any."""

    path: str
    entry: Any = None


def resolve_model(raw: str, config: str | None) -> Source | None:
    """The GGUF for ``raw``: the file itself, or a server-config model id
    or alias. None when neither matches. Refuses an entry that sets what
    gmlx kld does not apply."""
    path = os.path.expanduser(raw)
    if os.path.exists(path):
        return Source(os.path.abspath(path))
    if "/" in raw or os.sep in raw or raw.lower().endswith(".gguf"):
        return None
    import gmlx.config as cfgmod
    from gmlx.load.discovery import fill_families, merge_discovered

    try:
        cfg, _ = cfgmod.load_cli_config(config)
        if cfg is None:
            return None
        if raw.split("@", 1)[0] not in cfg.models and raw not in cfg.aliases:
            merge_discovered(cfg)
        fill_families(cfg)
        rm = cfgmod.resolve_cli_model(raw, cfg)
    except cfgmod.ConfigError as e:
        raise Refusal(str(e)) from None
    if rm is None:
        return None
    unsupported = [k for k, v in (("adapter", getattr(rm, "adapter_list", ()) or None),
                                  ("rope_scaling", getattr(rm, "rope_scaling", None)))
                   if v is not None]
    if unsupported:
        raise Refusal(f"{raw}: the config entry sets {' and '.join(unsupported)}, "
                      "which gmlx kld does not apply; pass the GGUF path")
    return Source(rm.path, rm)


def tokenize(tokenizer, text: str) -> tuple[np.ndarray, int | None]:
    """(ids, bos) for the text, as llama.cpp tokenizes it for
    llama-perplexity: control tokens written in the text stay text, and
    BOS goes in front only when the vocabulary adds it. ``bos`` is the id
    each chunk starts with, or None."""
    from gmlx.distill.tokens import adds_bos, bos_id
    from gmlx.load.tokenizer import backend

    b = backend(tokenizer)
    prev = b.encode_special_tokens
    b.encode_special_tokens = True
    try:
        ids = b.encode(text, add_special_tokens=False).ids
    finally:
        b.encode_special_tokens = prev
    bos = bos_id(tokenizer) if adds_bos(tokenizer) else None
    if bos is not None:
        ids = [bos, *ids]
    return np.asarray(ids, dtype=np.int32), bos


def tokenizer_mismatch(ref_tok, tok, ref_width: int, width: int,
                       ref_ids: np.ndarray, ids: np.ndarray) -> str | None:
    """Why two tokenizers cannot be compared on this text, or None."""
    from gmlx.distill.tokens import specials_digest
    from gmlx.load.tokenizer import hf_inner, vocab_map_hash

    if ref_width != width:
        return f"the output widths differ ({ref_width} vs {width})"
    if len(hf_inner(ref_tok)) != len(hf_inner(tok)):
        return (f"the vocabulary sizes differ ({len(hf_inner(ref_tok))} vs "
                f"{len(hf_inner(tok))})")
    if vocab_map_hash(ref_tok) != vocab_map_hash(tok):
        return "the vocabularies differ"
    if specials_digest(ref_tok) != specials_digest(tok):
        return "the special tokens differ"
    if ref_ids.shape != ids.shape or not np.array_equal(ref_ids, ids):
        return "the text tokenizes differently"
    return None


def rows_per_forward(n_scored: int, width: int, n_items: int,
                     budget: int | None = None) -> int:
    """Chunks per forward: as many as keep the float32 logits under
    ``budget`` (default ROW_BUDGET_BYTES), at least one, at most
    ``n_items``."""
    if budget is None:
        budget = ROW_BUDGET_BYTES
    per_row = 4 * n_scored * width
    return max(1, min(n_items, budget // max(per_row, 1)))


def slice_rows(width: int, budget: int | None = None) -> int:
    """Scored positions per head pass: as many as keep their float32
    logits under ``budget`` (default ROW_BUDGET_BYTES), at least one."""
    if budget is None:
        budget = ROW_BUDGET_BYTES
    return max(1, budget // (4 * width))


def _weight_block(mod, r0: int, r1: int):
    """Rows ``r0:r1`` of the head weight in float32, times its weight
    scale."""
    import mlx.core as mx

    from gmlx.distill.head import head_weight_rows
    from gmlx.load.out_scale import weight_scale

    if "scales" in mod:
        return head_weight_rows(mod, r0, r1, mx.float32)
    blk = mod["weight"][r0:r1].astype(mx.float32)
    s = weight_scale(mod)
    return blk if s is None else blk * s.astype(mx.float32)


class Scorer:
    """Float32 logits of a loaded text model at the scored positions of
    whole chunks. The trunk runs in the model's dtype; the head runs in
    float32 on the scored positions only."""

    def __init__(self, model, n_ctx: int):
        from gmlx.distill.head import head_spec_from_model

        self.inner = getattr(model, "language_model", model)
        self.spec = head_spec_from_model(self.inner)
        self.mod = self.spec.module
        self.width = int(self.spec.V)
        self.n_ctx = n_ctx
        self.first, self.n_scored = K.scored_span(n_ctx)

    def _project(self, h, out: np.ndarray) -> None:
        import mlx.core as mx

        h = h.astype(mx.float32)
        if self.spec.pre is not None:
            h = self.spec.pre(h)
        mx.eval(h)
        has_bias = "bias" in self.mod
        for r0 in range(0, self.width, HEAD_BLOCK):
            r1 = min(r0 + HEAD_BLOCK, self.width)
            z = h @ _weight_block(self.mod, r0, r1).T
            if has_bias:
                z = z + self.mod["bias"][r0:r1].astype(mx.float32)
            if self.spec.scale != 1.0:
                z = z * self.spec.scale
            if self.spec.softcap:
                z = self.spec.softcap * mx.tanh(z / self.spec.softcap)
            mx.eval(z)
            out[:, r0:r1] = np.asarray(z)

    def slices(self, ids: np.ndarray):
        """Float32 logits of the scored positions of chunk ids [rows, n_ctx],
        chunk-major, in slices [k, width] of at most slice_rows(width)
        positions. The trunk runs once."""
        import mlx.core as mx

        from gmlx.distill.student import trunk_hidden

        rows = int(ids.shape[0])
        h = trunk_hidden(self.inner, mx.array(ids))
        h = h[:, self.first:self.first + self.n_scored, :]
        h = h.reshape(rows * self.n_scored, h.shape[-1])
        mx.eval(h)
        step = slice_rows(self.width)
        for p0 in range(0, int(h.shape[0]), step):
            hs = h[p0:p0 + step]
            out = np.empty((int(hs.shape[0]), self.width), dtype=np.float32)
            self._project(hs, out)
            yield out

    def parity_gap(self, ids: np.ndarray) -> float:
        """The largest gap between this head and the model's own logits
        on ``ids`` [T], as a share of the largest logit."""
        import mlx.core as mx

        from gmlx.distill.student import trunk_hidden

        x = mx.array(ids[None])
        ref = self.inner(x)
        ref = np.array(getattr(ref, "logits", ref)[0].astype(mx.float32))
        h = trunk_hidden(self.inner, x)[0]
        mine = np.empty((h.shape[0], self.width), dtype=np.float32)
        self._project(h, mine)
        if ref.shape != mine.shape:
            return float("inf")
        return float(np.abs(mine - ref).max() / max(np.abs(ref).max(), 1.0))


@dataclass
class Plan:
    """What both passes share: the ids, the chunking and the widths."""

    tokens: np.ndarray
    bos: int | None
    n_ctx: int
    n_chunk: int
    width: int

    def groups(self, rows: int):
        for c0 in range(0, self.n_chunk, rows):
            yield list(range(c0, min(c0 + rows, self.n_chunk)))

    def ids(self, chunks: list[int]) -> np.ndarray:
        return np.stack([K.chunk_ids(self.tokens, c, self.n_ctx, self.bos)
                         for c in chunks])


Progress = Callable[[str, int, int], None]


def _slices(scorer: Scorer, plan: Plan, stage: str, progress: Progress):
    """Every scored position's logits in file order, in slices, with a
    progress call as each chunk completes."""
    rows = rows_per_forward(scorer.n_scored, plan.width, plan.n_chunk)
    done = reported = 0
    for chunks in plan.groups(rows):
        for z in scorer.slices(plan.ids(chunks)):
            yield z
            done += len(z)
            while reported < min(done // scorer.n_scored, plan.n_chunk):
                reported += 1
                progress(stage, reported, plan.n_chunk)


def score_reference(scorer: Scorer, plan: Plan, f, progress: Progress) -> None:
    """Write the reference file: the header, then each position's
    record."""
    K.write_reference_header(f, plan.n_ctx, plan.width, plan.n_chunk,
                             plan.tokens)
    for z in _slices(scorer, plan, "reference", progress):
        f.write(K.quantize_logprobs(z).tobytes())


def score_model(scorer: Scorer, plan: Plan, f, progress: Progress) -> K.KldSums:
    """KL sums of a model against the reference file ``f``."""
    n_ctx, width, n_chunk, tokens = K.read_reference_header(f)
    if (n_ctx, width, n_chunk) != (plan.n_ctx, plan.width, plan.n_chunk) \
            or not np.array_equal(tokens, plan.tokens[:n_ctx * n_chunk]):
        raise ValueError("the reference file does not match this run")
    sums = K.KldSums()
    for z in _slices(scorer, plan, "model", progress):
        sums.add(z, K.read_reference_records(f, len(z), width))
    return sums


def check_disk(directory: str, need: int) -> str | None:
    """Why ``need`` bytes do not fit in ``directory``, or None."""
    free = shutil.disk_usage(directory).free
    if free < need + DISK_MARGIN_BYTES:
        return (f"the reference log-probs need {need / 1e9:.2f} GB in "
                f"{directory}, which has {free / 1e9:.2f} GB free; set "
                "TMPDIR to a volume with more room or pass --chunks")
    return None


def _streams(ns) -> bool:
    return bool(getattr(ns, "stream_experts", False)
                or getattr(ns, "stream_cpu", False))


def _model_args(src: Source, args) -> argparse.Namespace:
    """The flags one model loads under: the command's, with its config
    entry's placement where no flag was typed."""
    from .cli import _apply_resolved_to_args

    ns = argparse.Namespace(**vars(args))
    if src.entry is not None:
        _apply_resolved_to_args(ns, src.entry, getattr(args, "_explicit", set()))
    ns.gguf = src.path
    return ns


def _load(src: Source, args) -> tuple[object, argparse.Namespace]:
    """Load one model. Returns it and the flags it loaded under."""
    import gmlx.load.loadlog as loadlog
    from gmlx.load.loader import load_model
    from gmlx.serve.cb_phase import relift

    from .cli import _apply_placement, _ensure_stream_cb_caps

    ns = _model_args(src, args)
    if _streams(ns):
        relift()
    _ensure_stream_cb_caps(ns)
    with loadlog.load_ui(args.verbose, src.path):
        model, _config, _tok = load_model(src.path)
    _apply_placement(ns, model)
    return model, ns


@contextlib.contextmanager
def _resident_wire(model, args):
    """The wired limit for a resident model on the GPU, put back after."""
    import mlx.core as mx

    from gmlx.device import cpu_default_device, metal_value
    from gmlx.distill.teacher import experts_streaming

    streamed = _streams(args) or experts_streaming(model)
    prev = None
    if not streamed and not cpu_default_device():
        try:
            prev = mx.set_wired_limit(
                int(metal_value("max_recommended_working_set_size")))
        except Exception:  # noqa: BLE001 - no device limit to set
            prev = None
    try:
        yield
    finally:
        if prev is not None:
            mx.set_wired_limit(prev)


def _release(model, path: str) -> None:
    """Close the model's streaming install (prefetcher, feeders, arena,
    weight pin) and sweep its page cache, before the next model loads."""
    from gmlx.stream.installs import release
    from gmlx.stream.pagecache import release_streaming_for

    release(model)
    release_streaming_for(path)


def _unload() -> None:
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _check_head(scorer: Scorer, plan: Plan, label: str) -> None:
    from gmlx.distill.head import HEAD_PARITY_TOL

    if scorer.width != plan.width:
        raise Refusal(f"{label}: the model's output width {scorer.width} "
                      f"is not the header's {plan.width}")
    probe = K.chunk_ids(plan.tokens, 0, plan.n_ctx, plan.bos)[:16]
    gap = scorer.parity_gap(probe)
    if not gap <= HEAD_PARITY_TOL:
        raise Refusal(
            f"{label}: the float32 head does not reproduce the model's own "
            f"logits (relative gap {gap:.3f}); this model changes its logits "
            "after the projection in a way gmlx kld does not carry")


def prepare(args) -> tuple[Source, Source, Plan]:
    """(MODEL, reference, plan), from GGUF headers only. Raises Refusal
    before any model loads."""
    from gmlx.distill.tokens import logits_width_from_gguf, tokenizer_from_gguf
    from gmlx.load.tokenizer import hf_inner

    if args.ctx < 3:
        raise Refusal("--ctx must be at least 3")
    if args.chunks is not None and args.chunks < 1:
        raise Refusal("--chunks must be at least 1")
    model = resolve_model(args.model, args.config)
    if model is None:
        raise Refusal(f"no such file or config model: {args.model}")
    ref = resolve_model(args.reference, args.config)
    if ref is None:
        raise Refusal(f"--reference: no such file or config model: "
                      f"{args.reference}")
    model_path, ref_path = model.path, ref.path
    try:
        text = K.read_text(os.path.expanduser(args.text))
    except (OSError, ValueError) as e:
        raise Refusal(f"--text: {e}") from None
    ref_tok = tokenizer_from_gguf(ref_path)
    tok = tokenizer_from_gguf(model_path)
    for label, t in (("--reference", ref_tok), ("MODEL", tok)):
        # llama-perplexity asserts the vocabulary adds no EOS.
        if getattr(hf_inner(t), "_gguf_add_eos_token", False) is True:
            raise Refusal(f"{label}: the vocabulary adds EOS to each text "
                          "(tokenizer.ggml.add_eos_token), which "
                          "llama-perplexity does not score")
    ref_ids, bos = tokenize(ref_tok, text)
    ids, _ = tokenize(tok, text)
    width = logits_width_from_gguf(ref_path)
    why = tokenizer_mismatch(ref_tok, tok, width,
                             logits_width_from_gguf(model_path), ref_ids, ids)
    if why:
        raise Refusal(f"the reference and MODEL tokenizers do not match: "
                      f"{why}")
    try:
        n_chunk = K.chunk_count(len(ref_ids), args.ctx, args.chunks)
    except ValueError as e:
        raise Refusal(str(e)) from None
    plan = Plan(tokens=ref_ids, bos=bos, n_ctx=args.ctx, n_chunk=n_chunk,
                width=width)
    why = check_disk(tempfile.gettempdir(),
                     K.reference_bytes(args.ctx, width, n_chunk))
    if why:
        raise Refusal(why)
    return model, ref, plan


def run(args, emit: Callable[[dict], None], log: Callable[[str], None]) -> dict:
    """Both passes. Returns the result record; raises Refusal when the
    command refuses the run."""
    model_src, ref_src, plan = prepare(args)
    model_path, ref_path = model_src.path, ref_src.path
    n_chunk = plan.n_chunk
    tmpdir = tempfile.gettempdir()

    def progress(stage: str, chunk: int, total: int) -> None:
        emit({"event": "progress", "stage": stage, "chunk": chunk,
              "chunks": total})

    fd, ref_file = tempfile.mkstemp(prefix="gmlx-kld-", suffix=".logp",
                                    dir=tmpdir)
    os.close(fd)
    try:
        log(f"[kld] reference {os.path.basename(ref_path)}: {n_chunk} chunks "
            f"of {args.ctx} tokens")
        model, ns = _load(ref_src, args)
        try:
            scorer = Scorer(model, args.ctx)
            _check_head(scorer, plan, "--reference")
            with _resident_wire(model, ns), open(ref_file, "wb") as f:
                score_reference(scorer, plan, f, progress)
        finally:
            scorer = None
            _release(model, ref_path)
            model = None
            _unload()
        log(f"[kld] model {os.path.basename(model_path)}")
        model, ns = _load(model_src, args)
        try:
            scorer = Scorer(model, args.ctx)
            _check_head(scorer, plan, "MODEL")
            with _resident_wire(model, ns), open(ref_file, "rb") as f:
                sums = score_model(scorer, plan, f, progress)
        finally:
            scorer = None
            _release(model, model_path)
            model = None
            _unload()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(ref_file)
    mean, mean_unc = sums.mean_kld()
    top, top_unc = sums.same_top()
    return {"event": "result", "model": model_path, "reference": ref_path,
            "adapter": None, "strength": None, "mean_kld": mean,
            "mean_kld_unc": mean_unc, "same_top_p": top,
            "same_top_p_unc": top_unc, "positions": sums.count,
            "chunks": n_chunk, "ctx": args.ctx}


def format_table(result: dict) -> str:
    return "\n".join([
        f"model:      {result['model']}",
        f"reference:  {result['reference']}",
        f"scored:     {result['positions']} positions in {result['chunks']} "
        f"chunks of {result['ctx']} tokens",
        "",
        f"{'point':<8}{'Mean KLD':<23}Same top p",
        f"{'MODEL':<8}{result['mean_kld']:.6f} +/- {result['mean_kld_unc']:.6f}"
        f"  {100 * result['same_top_p']:.3f} +/- "
        f"{100 * result['same_top_p_unc']:.3f} %",
    ])


def parse_args(argv: list[str], prog: str = "gmlx kld") -> argparse.Namespace:
    """The parsed flags, with the set the user passed in ``_explicit``, so
    a config entry never overrides a typed flag."""
    from .cli import _explicit_dests

    parser = build_parser(prog)
    args = parser.parse_args(argv)
    args._explicit = _explicit_dests(parser, argv)
    return args


def cmd_kld(argv: list[str], prog: str = "gmlx kld") -> int:
    args = parse_args(argv, prog)
    if not args.reference:
        print(f"error: {prog} needs --reference FILE, the GGUF to score "
              "MODEL against", file=sys.stderr)
        return 2
    out = sys.stdout

    def emit(rec: dict) -> None:
        if args.json:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        elif rec["event"] == "progress":
            print(f"[kld] {rec['stage']}: chunk {rec['chunk']}/"
                  f"{rec['chunks']}", file=sys.stderr, flush=True)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # In JSON mode stdout carries only the JSON lines.
    redirect = (contextlib.redirect_stdout(sys.stderr) if args.json
                else contextlib.nullcontext())
    try:
        with redirect:
            result = run(args, emit, log)
    except Refusal as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        emit(result)
    else:
        print(format_table(result))
    return 0
