"""``gmlx distill``: offline distillation in six steps plus one
diagnostic. ``gen`` runs a
teacher through ``gmlx serve`` over a prompt set and writes its replies as
a corpus, ``filter`` drops the rows a student should not learn from,
``cache`` runs the teacher once over a corpus and stores its top-k
log-probabilities, ``align`` maps that cache onto a student tokenizer, ``train``
fits a LoRA adapter on a K-quant GGUF student against the view, and
``eval`` scores the student before and after. ``census`` measures, from
two caches of the same replies, how much a context moves the teacher.
The library behind each action is ``gmlx.distill``."""
from __future__ import annotations

import argparse
import sys

_ACTIONS = ("gen", "filter", "cache", "align", "train", "eval", "census")
_ACTION_DESC = {
    "gen": "run a teacher over a prompt set through gmlx serve and write a corpus",
    "filter": "drop generated rows a student should not learn from",
    "cache": "run a teacher over a corpus and store its top-k log-probabilities",
    "align": "map a cache onto a student tokenizer and write a view",
    "train": "train a LoRA adapter on a GGUF student against a view",
    "eval": "score a student, with and without its adapter, on held-out data",
    "census": "measure how much a context moves the teacher between two caches of the same replies",
}


def _print_help(prog: str) -> None:
    lines = [f"usage: {prog} <action> [options]", "",
             "offline distillation: generate or bring a corpus, cache a teacher once, align",
             "the cache to a student, train against the view, evaluate the result.", "", "actions:"]
    for a in _ACTIONS:
        lines.append(f"  {a:<8} {_ACTION_DESC[a]}")
    lines += ["", f"run `{prog} <action> --help` for an action's options.",
              "every size flag is in decimal GB (1e9 bytes)."]
    print("\n".join(lines))


def _gen_parser(prog: str) -> argparse.ArgumentParser:
    from gmlx.distill.gen import DEFAULT_CONTEXT_FORMAT
    from gmlx.distill.teacher import CONTINUE_INSTRUCTION
    p = argparse.ArgumentParser(
        prog=prog,
        description="Run a teacher to its own end of turn over a prompt set through gmlx serve, with "
                    "request fan-out, and write the replies as a conversation corpus with a generator "
                    "sidecar. The run resumes, since prompt ids already in --out are skipped.")
    p.add_argument("--out", required=True, metavar="PATH", help="Corpus jsonl to write, with <out>.gen.json beside it.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompts", metavar="PATH",
                     help="A jsonl of {id, messages, context?} rows whose messages end on a user turn.")
    src.add_argument("--corpus", metavar="PATH|ID",
                     help="A text corpus (jsonl, directory or Hugging Face id) to build continuation prompts from.")
    p.add_argument("--teacher", metavar="GGUF", help="Teacher GGUF to serve for the run.")
    p.add_argument("--base-url", default=None, metavar="URL",
                   help="A running server's /v1 base to use instead of serving --teacher.")
    p.add_argument("--host", default="127.0.0.1", help="Bind host of the served teacher (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=8093, help="Port of the served teacher (default 8093).")
    p.add_argument("--text-key", default="text", help="With --corpus: text column of a jsonl or dataset row.")
    p.add_argument("--hf-split", default="train", help="With --corpus: dataset split for a Hugging Face id.")
    p.add_argument("--prefix-chars", type=int, default=1500,
                   help="With --corpus: document prefix quoted in the user turn (default 1500).")
    p.add_argument("--min-chars", type=int, default=2000,
                   help="With --corpus: skip documents shorter than this (default 2000).")
    p.add_argument("--docs", type=int, default=0, help="With --corpus: prompts to build (default all).")
    p.add_argument("--instruction", default=CONTINUE_INSTRUCTION,
                   help="With --corpus: the user turn placed before the document prefix.")
    p.add_argument("--chat-template-kwargs", default=None, metavar="JSON",
                   help="Passed to gmlx serve --chat-template-config: the teacher's render settings.")
    p.add_argument("--context", default=None, metavar="FILE",
                   help="Text the teacher reads for every prompt without its own context field. The "
                        "student's list is written without it.")
    p.add_argument("--context-format", default=DEFAULT_CONTEXT_FORMAT,
                   help="How the context and the last user turn combine for the teacher.")
    p.add_argument("--thinking", action="store_true",
                   help="Turn the teacher's reasoning on. The reasoning trace is kept as reasoning_content on the reply.")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="With --thinking, cap the reasoning trace at this many tokens per request. The trace is "
                        "counted with the teacher's tokenizer to mark the replies it cut.")
    p.add_argument("--tokenizer", default=None, metavar="GGUF|DIR",
                   help="Tokenizer for the reasoning trace count with --base-url (default: read from --teacher).")
    p.add_argument("--serve-arg", action="append", default=[], metavar="ARG",
                   help="Extra gmlx serve argument, repeatable.")
    p.add_argument("--startup-timeout", type=float, default=900.0,
                   help="Seconds to wait for the served teacher (default 900).")
    p.add_argument("--concurrency", type=int, default=8, help="Requests in flight (default 8).")
    p.add_argument("--max-tokens", type=int, default=1024, help="Reply budget per request, not counting the reasoning trace (default 1024).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default 0.7).")
    p.add_argument("--top-p", type=float, default=0.9, help="Keep the most likely tokens whose probabilities add to this (default 0.9).")
    p.add_argument("--top-k", type=int, default=None, help="Top-k cutoff (default the server's).")
    p.add_argument("--min-p", type=float, default=None, help="Minimum-probability cutoff (default the server's).")
    p.add_argument("--seed", type=int, default=1,
                   help="Base seed, and each request uses it plus the prompt index (default 1).")
    p.add_argument("--timeout", type=float, default=1800.0, help="Per-request timeout in seconds (default 1800).")
    p.add_argument("--report-every", type=int, default=50, help="Progress line interval in replies (default 50).")
    return p


def _filter_parser(prog: str) -> argparse.ArgumentParser:
    from gmlx.distill.gen import DEFAULT_CONTEXT_FORMAT
    p = argparse.ArgumentParser(
        prog=prog,
        description="Drop generated rows a student should not learn from, in a fixed order of checks, "
                    "and stamp the filter version into the corpus sidecar. Optionally run a task-specific "
                    "verify command over the survivors, or put a context the replier never saw on the "
                    "teacher's side of every kept row.")
    p.add_argument("--in", dest="inputs", action="append", required=True, metavar="PATH",
                   help="Generated corpus jsonl, repeatable and concatenated in order.")
    p.add_argument("--out", required=True, metavar="PATH", help="Filtered corpus to write, with <out>.gen.json beside it.")
    p.add_argument("--report", default=None, metavar="JSON", help="Write the kept and dropped counts here.")
    p.add_argument("--rejects", default=None, metavar="PATH",
                   help="Write one {id, reason} line per dropped row here.")
    p.add_argument("--min-tokens", type=int, default=16,
                   help="Drop replies with fewer whitespace tokens than this (default 16).")
    p.add_argument("--ngram", type=int, default=8, help="N-gram size of the repetition check (default 8).")
    p.add_argument("--max-repeat", type=float, default=0.2,
                   help="Drop replies whose repeated n-grams exceed this fraction (default 0.2).")
    p.add_argument("--max-line-repeats", type=int, default=2,
                   help="Drop replies with a line repeated more than this many times in a row (default 2).")
    p.add_argument("--max-non-ascii", type=float, default=None,
                   help="Drop replies whose non-ASCII character fraction exceeds this (default off).")
    p.add_argument("--max-reply-tokens", type=int, default=None,
                   help="Drop replies longer than this many completion tokens (default off).")
    p.add_argument("--keep-budget-hit", action="store_true",
                   help="Keep replies whose thinking budget cut the reasoning trace (dropped by default).")
    p.add_argument("--verify", default=None, metavar="CMD",
                   help="Shell command that reads the surviving rows as jsonl on stdin and prints one line "
                        "per row: ok, or a reason word to drop it.")
    p.add_argument("--context", default=None, metavar="FILE",
                   help="Put this text on the teacher's side of every kept row, keeping the prompt as "
                        "given under student_messages.")
    p.add_argument("--context-format", default=DEFAULT_CONTEXT_FORMAT,
                   help="How the context and the last user turn combine for the teacher.")
    return p


def _cache_parser(prog: str) -> argparse.ArgumentParser:
    from gmlx.distill.teacher import CONTINUE_INSTRUCTION, FRAME_CHOICES
    p = argparse.ArgumentParser(
        prog=prog,
        description="Run a teacher GGUF over a corpus once and store, per position, its top-k "
                    "log-probabilities and the side fields a student of any tokenizer needs. Sizes in decimal GB.")
    p.add_argument("--teacher", metavar="GGUF", help="Teacher GGUF (sharded ok).")
    p.add_argument("--corpus", metavar="PATH|ID",
                   help="A jsonl file, a directory of text files, or a Hugging Face dataset id "
                        "(id[@config], which needs the datasets package).")
    p.add_argument("--out", metavar="DIR", help="Cache directory to write.")
    p.add_argument("--validate", metavar="DIR", help="Validate an existing cache and exit.")
    p.add_argument("--top-k", type=int, default=256, help="Log-probabilities kept per position (default 256).")
    p.add_argument("--max-len", type=int, default=2048,
                   help="Teacher tokens per window including the start token (default 2048).")
    p.add_argument("--max-disk-gb", type=float, default=None,
                   help="Refuse when the size estimate exceeds this (default none).")
    p.add_argument("--cache-limit-gb", type=float, default=8.0,
                   help="MLX buffer cache cap during the pass (default 8).")
    p.add_argument("--logits-cap-gb", type=float, default=4.0,
                   help="Memory cap that sizes the head sub-chunk (default 4).")
    p.add_argument("--floor", action="store_true",
                   help="Also store floor_kld, the KL against the f16-rounded top-k.")
    p.add_argument("--rows-per-shard", type=int, default=64, help="Rows per shard file (default 64).")
    p.add_argument("--trunk", type=int, default=None,
                   help="Trunk chunk in tokens (default 512 for a teacher that fits in memory, 8192 streaming).")
    p.add_argument("--resume", action="store_true", help="Continue after the last verified shard.")
    p.add_argument("--max-rows", type=int, default=None, help="Stop after this many rows.")
    p.add_argument("--max-tokens", type=int, default=None, help="Stop after this many teacher tokens.")
    p.add_argument("--limit-docs", type=int, default=None, help="Read at most this many documents.")
    p.add_argument("--text-key", default="text", help="Text column of a jsonl or dataset row (default text).")
    p.add_argument("--hf-split", default="train", help="Dataset split for a Hugging Face id (default train).")
    p.add_argument("--source", default=None,
                   help="Source tag for every row (default human, or synthetic when <corpus>.gen.json exists).")
    p.add_argument("--frame", choices=FRAME_CHOICES, default="none",
                   help="none: raw text rows. continue: each window as a model turn behind "
                        "--frame-instruction. chat: rows are conversations, targets on every assistant "
                        "turn. reply: targets on the final assistant turn only. reply-think: a reply row "
                        "whose target starts at the final turn's reasoning content.")
    p.add_argument("--per-turn", action="store_true",
                   help="With --frame chat or reply: one reply row per assistant turn with the history "
                        "before it.")
    p.add_argument("--student-messages-key", default="student_messages",
                   help="Corpus key of the student's own message list on reply rows "
                        "(default student_messages).")
    p.add_argument("--frame-instruction", default=CONTINUE_INSTRUCTION,
                   help="User turn for --frame continue.")
    p.add_argument("--messages-key", default="messages",
                   help="Conversation column for the chat and reply frames (default messages).")
    p.add_argument("--close-final-windows", action="store_true",
                   help="With --frame continue: the window that ends its document is closed by the "
                        "template's turn-end marker, which becomes a target.")
    p.add_argument("--frame-kwargs", default=None, metavar="JSON",
                   help="Chat-template kwargs for every teacher render, as a JSON object or a file.")
    p.add_argument("--hf-source", default=None, metavar="ID", help="Tokenizer and config fallback.")
    p.add_argument("--no-require-feeder", dest="require_feeder", action="store_false",
                   help="Run a streaming teacher without the prefill feeder (every expert byte is "
                        "then read through the page cache).")
    p.add_argument("--no-wired-limit", action="store_true",
                   help="Leave the wired limit where it is for a teacher that fits in memory.")
    p.add_argument("--stream-experts", action="store_true",
                   help="Force expert streaming on a MoE teacher that would fit in memory.")
    p.add_argument("--expert-bytes-gb", type=float, default=None,
                   help="Expert bytes read per forward, for the read-traffic report of a streaming teacher.")
    p.add_argument("--routes", action="store_true",
                   help="MoE teachers: store every layer's top-k expert ids per position (the routes field, "
                        "uint8 up to 256 experts) and a routing block in the manifest, for replay by eval.")
    p.add_argument("--hidden", action="store_true",
                   help="Also store a seeded random sketch of the teacher's final hidden state per position "
                        "(the hidden field, float16), for train --hs.")
    p.add_argument("--hidden-dim", type=int, default=256, help="Width of the hidden sketch (default 256).")
    p.add_argument("--hidden-seed", type=int, default=1, help="Seed of the sketch matrix (default 1).")
    p.add_argument("--cpu", action="store_true", help="Run on the CPU device (smoke tests).")
    return p


def _align_parser(prog: str) -> argparse.ArgumentParser:
    from gmlx.distill.constants import DEFAULT_KNOBS
    p = argparse.ArgumentParser(
        prog=prog,
        description="Map a cache onto a student tokenizer in one CPU pass: tables for the tokenizer pair, "
                    "the projection census, the train and validation row index, and optionally the "
                    "materialized batch tensors.")
    p.add_argument("--cache", required=True, metavar="DIR", help="Cache directory from `distill cache`.")
    p.add_argument("--student", required=True, metavar="GGUF|DIR",
                   help="Student GGUF, or an MLX checkpoint directory for its tokenizer.")
    p.add_argument("--out", required=True, metavar="DIR", help="View directory to write.")
    p.add_argument("--tables", default=None, metavar="DIR",
                   help="An earlier view's tables.safetensors to reuse when the tokenizer pair matches.")
    p.add_argument("--kprime", type=int, default=None,
                   help="Cap on distinct student-token groups kept per boundary (default: the maximum seen).")
    p.add_argument("--materialize", action="store_true",
                   help="Also write the batch tensors as view shards, for a pair whose loader is slow.")
    p.add_argument("--max-disk-gb", type=float, default=None,
                   help="Refuse to materialize past this size (default none).")
    p.add_argument("--force", action="store_true", help="Keep a view the own-group check would refuse.")
    p.add_argument("--val-fraction", type=float, default=0.02,
                   help="Fraction of rows held for validation (default 0.02).")
    p.add_argument("--seed", type=int, default=1, help="Seed of the validation split (default 1).")
    p.add_argument("--w-mid", type=float, default=DEFAULT_KNOBS["w_mid"],
                   help="Weight of an intra-word shared boundary (default 0.5).")
    p.add_argument("--gamma", type=float, default=DEFAULT_KNOBS["gamma"],
                   help="Drop chunks of the ALM term, the cross-tokenizer chunk term, whose teacher boundary mass "
                        "is below this (default 0.001).")
    p.add_argument("--tau-alm", type=float, default=DEFAULT_KNOBS["tau_alm"],
                   help="Temperature on the ALM term (default 1.0).")
    p.add_argument("--T-dk", type=float, default=DEFAULT_KNOBS["T_dk"],
                   help="Temperature on the conditional factor of the bucketed KL (default 1.0).")
    p.add_argument("--max-chunk-len", type=int, default=DEFAULT_KNOBS["max_chunk_len"],
                   help="Longest ALM chunk in tokens on either side (default 8).")
    p.add_argument("--frame-kwargs", default=None, metavar="JSON",
                   help="Chat-template kwargs for every student render, stored in the view.")
    p.add_argument("--cpu", action="store_true", help="Run on the CPU device (smoke tests).")
    return p


def _train_parser(prog: str) -> argparse.ArgumentParser:
    from gmlx.distill.constants import DEFAULT_KNOBS
    p = argparse.ArgumentParser(
        prog=prog,
        description="Train a LoRA adapter on a K-quant GGUF student against one or more views and write "
                    "it as a GGUF adapter. The loop is gmlx's own: seeded batch order, resume by exact "
                    "iteration, last and best checkpoints.")
    p.add_argument("--view", action="append", default=[], required=True, metavar="DIR",
                   help="View directory from `distill align`, repeatable to mix views over one tokenizer pair.")
    p.add_argument("--student", required=True, metavar="GGUF", help="Student GGUF (sharded ok).")
    p.add_argument("--adapter-out", required=True, metavar="PATH", help="Output path for the .gguf adapter.")
    p.add_argument("--iters", type=int, required=True, help="Training iterations.")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default 16).")
    p.add_argument("--lora-scale", type=float, default=None,
                   help="LoRA multiplier as is (default 2.0 unless --lora-alpha is given).")
    p.add_argument("--lora-alpha", type=float, default=None,
                   help="LoRA multiplier as alpha / rank. Give this or --lora-scale, not both.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (default 0.0).")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recompute each layer's activations in the backward pass.")
    p.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (default 1e-4).")
    p.add_argument("--batch-size", type=int, default=8, help="Rows per step (default 8).")
    p.add_argument("--warmup", type=float, default=0.05,
                   help="Warmup as a fraction of the iterations, then cosine decay (default 0.05).")
    p.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default 0 for LoRA).")
    p.add_argument("--clip", type=float, default=1.0, help="Gradient norm clip (default 1.0).")
    p.add_argument("--seed", type=int, default=1, help="Data order and LoRA init (default 1).")
    p.add_argument("--loss", choices=["bucketed", "paper", "renorm"], default="bucketed",
                   help="bucketed: sparse KL with the tail bucket. paper: the top-k term with no tail bucket. "
                        "renorm: softmax over the support only.")
    p.add_argument("--dk", type=float, default=DEFAULT_KNOBS["lambda_dk"],
                   help="Weight of the bucketed KL term (default 1).")
    p.add_argument("--alm", type=float, default=DEFAULT_KNOBS["lambda_alm"],
                   help="Weight of the ALM term, 0 on a same-tokenizer view (default 1).")
    p.add_argument("--ce", type=float, default=DEFAULT_KNOBS["lambda_ce"],
                   help="Weight of the cross-entropy term (default 0).")
    p.add_argument("--T-dk", type=float, default=None, help="Override the view's T_dk.")
    p.add_argument("--tau-alm", type=float, default=None, help="Override the view's tau_alm.")
    p.add_argument("--gamma", type=float, default=None, help="Override the view's gamma.")
    p.add_argument("--chunk", type=int, default=512, help="Positions per head chunk (default 512).")
    p.add_argument("--hs", type=float, default=0.0,
                   help="Weight of the hidden-state term, a learned linear map from the student's final "
                        "hidden state to the cache's sketch at every boundary (default 0, off).")
    p.add_argument("--hs-loss", choices=("cosine", "mse"), default="cosine",
                   help="cosine: 1 - cosine similarity. mse: squared error on unit vectors (default cosine).")
    p.add_argument("--ckpt-dir", default=None, metavar="DIR",
                   help="Checkpoint directory (default ./ckpt in the working directory).")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint.")
    p.add_argument("--save-every", type=int, default=200, help="Checkpoint interval in steps (default 200).")
    p.add_argument("--val-every", type=int, default=200, help="Validation interval in steps (default 200).")
    p.add_argument("--val-batches", type=int, default=16, help="Validation batches per pass (default 16).")
    p.add_argument("--report-every", type=int, default=10, help="Train-loss report interval (default 10).")
    p.add_argument("--report", default=None, metavar="JSON", help="Write the run log here.")
    p.add_argument("--hf-source", default=None, metavar="ID", help="Tokenizer and config fallback.")
    p.add_argument("--no-wired-limit", action="store_true", help="Leave the wired limit where it is.")
    p.add_argument("--cache-limit-gb", type=float, default=8.0, help="MLX buffer cache cap (default 8).")
    p.add_argument("--cpu", action="store_true", help="Run on the CPU device (smoke tests).")
    return p


def _eval_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Score a GGUF student, with and without its adapter in one process: bits per byte on "
                    "held-out slices, sparse KL against a same-tokenizer cache, downstream tasks from "
                    "local jsonl files, and a chat sanity set. Writes a Markdown and a JSON report.")
    p.add_argument("--student", required=True, metavar="GGUF", help="Student GGUF.")
    p.add_argument("--adapter", default=None, metavar="GGUF", help="GGUF adapter to apply.")
    p.add_argument("--md", required=True, metavar="PATH", help="Markdown report to write.")
    p.add_argument("--json", required=True, metavar="PATH", help="JSON report to write.")
    p.add_argument("--cache", default=None, metavar="DIR",
                   help="Cache whose corpus the slices are decontaminated against.")
    p.add_argument("--slice", action="append", default=[], metavar="NAME=PATH",
                   help="A held-out text slice, repeatable.")
    p.add_argument("--teacher-bpb", default=None, metavar="JSON", help="Teacher bits per byte per slice.")
    p.add_argument("--tasks-dir", default=None, metavar="DIR",
                   help="Directory of task files: arc_easy.jsonl, hellaswag.jsonl, gsm8k.jsonl, gsm8k_shots.jsonl "
                        "(default the working directory).")
    p.add_argument("--tasks", default="", help="Comma list of arc_easy, hellaswag, gsm8k.")
    p.add_argument("--task-limit", type=int, default=None, help="Items per task (default all).")
    p.add_argument("--gsm8k-max-tokens", type=int, default=384, help="Generation budget per GSM8K item (default 384).")
    p.add_argument("--before", action="store_true",
                   help="Also score with the adapter disabled in process.")
    p.add_argument("--chat-slice", action="append", default=[], metavar="NAME=PATH",
                   help="A jsonl of {messages} conversations scored on their assistant turns, repeatable.")
    p.add_argument("--chat-sanity", default=None, metavar="PATH",
                   help="A jsonl of {id, messages, kind} chat prompts scored for template compliance and drift, "
                        "how far the replies moved from an earlier report's.")
    p.add_argument("--chat-max-tokens", type=int, default=256, help="Reply budget for the chat sanity set (default 256).")
    p.add_argument("--chat-refs", default=None, metavar="JSON",
                   help="An earlier eval report whose replies anchor the drift score.")
    p.add_argument("--chat-max-len", type=int, default=2048, help="Longest conversation scored (default 2048).")
    p.add_argument("--chat-per-turn", action="store_true", help="Score every assistant turn as its own row.")
    p.add_argument("--reply-slice", action="append", default=[], metavar="NAME=PATH",
                   help="A jsonl of conversations scored on the final reply, repeatable.")
    p.add_argument("--reply-think", action="store_true",
                   help="Reply slices target the final turn's reasoning content.")
    p.add_argument("--reply-positions", default=None, metavar="JSON",
                   help="A distill census JSON whose high_delta map restricts every reply slice to the "
                        "positions the context moved.")
    p.add_argument("--kld-cache", default=None, metavar="DIR",
                   help="Same-tokenizer cache to score sparse KL against.")
    p.add_argument("--kld-rows", type=int, default=None, help="Rows of the KL cache to score (default all).")
    p.add_argument("--frame-kwargs", default=None, metavar="JSON",
                   help="Chat-template kwargs for every render, as a JSON object or a file.")
    p.add_argument("--max-len", type=int, default=512, help="Window length for bits per byte (default 512).")
    p.add_argument("--bpb-prefix", default=None,
                   help="Text placed before every window, or @KIND for a frame prefix.")
    p.add_argument("--batch-size", type=int, default=8, help="Windows per batch (default 8).")
    p.add_argument("--cache-limit-gb", type=float, default=4.0, help="MLX buffer cache cap (default 4).")
    p.add_argument("--decontam-threshold", type=float, default=0.01,
                   help="Slice window fraction found in the corpus above which its gate is void (default 0.01).")
    p.add_argument("--hf-source", default=None, metavar="ID", help="Tokenizer and config fallback.")
    p.add_argument("--cpu", action="store_true", help="Run on the CPU device (smoke tests).")
    return p


def _census_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Measure how much a context the student never sees moves the teacher, from two or "
                    "more reply caches (distill cache --frame reply) of the same prompts: one cut "
                    "without the context and one per context. Writes a JSON whose high_delta map "
                    "distill eval --reply-positions reads, and a Markdown summary. CPU only.")
    p.add_argument("--without", required=True, metavar="DIR", help="Cache of the prompts without any context.")
    p.add_argument("--with", dest="with_", action="append", required=True, metavar="DIR",
                   help="Cache with a context, repeatable.")
    p.add_argument("--out", required=True, metavar="JSON", help="Census JSON to write.")
    p.add_argument("--md", default=None, metavar="PATH", help="Markdown summary to write.")
    p.add_argument("--corpus", default=None, metavar="JSONL",
                   help="The prompts jsonl the caches were made from, so high_delta is keyed by row id.")
    p.add_argument("--delta-threshold", type=float, default=1.0,
                   help="Nats gained at the token the teacher wrote that make a position high-delta (default 1.0).")
    p.add_argument("--pair-by", choices=("line", "doc"), default="line",
                   help="Pair rows across caches by corpus line (default) or by the full doc_id.")
    p.add_argument("--max-rows", type=int, default=None, help="Paired rows to measure (default all).")
    return p


PARSERS = {"gen": _gen_parser, "filter": _filter_parser, "cache": _cache_parser, "align": _align_parser,
           "census": _census_parser,
           "train": _train_parser, "eval": _eval_parser}


def _cpu(args) -> None:
    if getattr(args, "cpu", False):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)


def cmd_gen(argv: list[str], prog: str = "gmlx distill gen") -> int:
    p = _gen_parser(prog)
    args = p.parse_args(argv)
    from gmlx.distill.gen import GenOptions, run_gen
    fields = {k: v for k, v in vars(args).items() if k in GenOptions.__dataclass_fields__}
    return run_gen(GenOptions(**fields))


def cmd_filter(argv: list[str], prog: str = "gmlx distill filter") -> int:
    args = _filter_parser(prog).parse_args(argv)
    from gmlx.distill.filter import FilterOptions, run_filter
    fields = {k: v for k, v in vars(args).items() if k in FilterOptions.__dataclass_fields__}
    return run_filter(FilterOptions(**fields))


def cmd_cache(argv: list[str], prog: str = "gmlx distill cache") -> int:
    p = _cache_parser(prog)
    args = p.parse_args(argv)
    _cpu(args)
    if args.validate:
        from pathlib import Path

        from gmlx.distill.format import validate_cache
        problems = validate_cache(Path(args.validate))
        print("valid" if not problems else "\n".join("[cache] validate: " + x for x in problems))
        return 0 if not problems else 1
    if not (args.teacher and args.corpus and args.out):
        print("[cache] refuse: --teacher, --corpus and --out are required, or --validate DIR", file=sys.stderr)
        return 2
    from gmlx.distill.teacher import CacheOptions, run_cache
    fields = {k: v for k, v in vars(args).items() if k in CacheOptions.__dataclass_fields__}
    return run_cache(CacheOptions(**fields))


def cmd_align(argv: list[str], prog: str = "gmlx distill align") -> int:
    args = _align_parser(prog).parse_args(argv)
    _cpu(args)
    from gmlx.distill.view import AlignOptions, run_align
    fields = {k: v for k, v in vars(args).items() if k in AlignOptions.__dataclass_fields__}
    return run_align(AlignOptions(**fields))


def cmd_train(argv: list[str], prog: str = "gmlx distill train") -> int:
    p = _train_parser(prog)
    args = p.parse_args(argv)
    _cpu(args)
    from gmlx.distill.trainer import TrainOptions, run_train
    fields = {k: v for k, v in vars(args).items() if k in TrainOptions.__dataclass_fields__}
    fields["views"] = args.view
    return run_train(TrainOptions(**fields))


def cmd_eval(argv: list[str], prog: str = "gmlx distill eval") -> int:
    args = _eval_parser(prog).parse_args(argv)
    _cpu(args)
    from gmlx.distill.evaluate import EvalOptions, run_eval
    fields = {k: v for k, v in vars(args).items() if k in EvalOptions.__dataclass_fields__}
    fields["slices"] = args.slice
    fields["chat_slices"] = args.chat_slice
    fields["reply_slices"] = args.reply_slice
    return run_eval(EvalOptions(**fields))


def cmd_census(argv: list[str], prog: str = "gmlx distill census") -> int:
    args = _census_parser(prog).parse_args(argv)
    from gmlx.distill.census import CensusOptions, run_census
    return run_census(CensusOptions(**vars(args)))


HANDLERS = {"gen": cmd_gen, "filter": cmd_filter, "cache": cmd_cache, "align": cmd_align, "train": cmd_train,
            "eval": cmd_eval, "census": cmd_census}


def cmd_distill(argv: list[str], prog: str = "gmlx distill") -> int:
    """Dispatch ``gmlx distill <action> ...``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(prog)
        return 0
    action, rest = argv[0], argv[1:]
    if action not in HANDLERS:
        print(f"[distill] refuse: unknown action {action!r}, one of {', '.join(_ACTIONS)}", file=sys.stderr)
        return 2
    return HANDLERS[action](rest, prog=f"{prog} {action}")
