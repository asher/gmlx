"""docs/distill.md walks: every ``gmlx distill`` command in the guide parses
against its action's parser, every file a command consumes was produced by
an earlier command or is one the guide tells the reader to write, every
``./script.py`` has its ``chmod +x``, and every quoted ``[action] ...`` log
string exists in the code. No model loads."""
from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from gmlx.commands import distill as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "distill.md").read_text(encoding="utf-8")
CODE = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "gmlx" / "distill").glob("*.py"))
CODE += (ROOT / "gmlx" / "commands" / "distill.py").read_text(encoding="utf-8")
PARSERS = {"gen": D._gen_parser, "filter": D._filter_parser, "cache": D._cache_parser, "align": D._align_parser,
           "train": D._train_parser, "eval": D._eval_parser, "census": D._census_parser}
# files the guide tells the reader to write or supply
USER_WRITTEN = {"schema.md", "freight.sqlite", "check-sql.py", "judge.py", "make-prompts.py", "chat-sanity.jsonl",
                "prompts-train.jsonl", "prompts-heldout.jsonl", "prompts-untrained.jsonl",
                "prompts-heldout-combined.jsonl", "heldout-prose.txt", "heldout-code.txt", "corpus.jsonl",
                "teacher-Q6_K.gguf", "student-Q4_K_M.gguf"}
PRODUCING = ("out", "adapter_out", "md", "json", "report", "rejects")
CONSUMING = ("teacher", "student", "adapter", "corpus", "prompts", "context", "cache", "tables",
             "reply_positions", "kld_cache", "without", "chat_sanity")
CONSUMING_LISTS = ("inputs", "views", "with_", "slices", "reply_slices")


def _commands():
    for block in re.findall(r"```sh\n(.*?)```", GUIDE, re.S):
        for line in block.replace("\\\n", " ").split("\n"):
            line = line.strip()
            if line.startswith(("gmlx ", "python3 ")):
                yield shlex.split(line)


def _parse(action, rest):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        try:
            return vars(PARSERS[action]("gmlx distill " + action).parse_args(rest)), ""
        except SystemExit:
            return None, buf.getvalue().strip().splitlines()[-1]


def test_every_guide_command_parses_and_its_inputs_exist():
    produced, problems = set(), []
    for argv in _commands():
        if argv[:2] == ["gmlx", "pull"]:
            produced.add(argv[2].split("/")[-1])
            continue
        if argv[0] == "python3" and ">" in argv:
            produced.add(argv[argv.index(">") + 1])
            continue
        if argv[:2] != ["gmlx", "distill"]:
            continue
        action, rest = argv[2], argv[3:]
        d, err = _parse(action, rest)
        if d is None:
            problems.append(f"{action}: {err}: {' '.join(rest)[:80]}")
            continue
        consumed = [Path(d[k]).name for k in CONSUMING if d.get(k)]
        consumed += [Path(v.split("=", 1)[-1]).name for k in CONSUMING_LISTS for v in (d.get(k) or [])]
        consumed += [a.split("=", 1)[1] for a in rest if a.startswith("--serve-arg=") and a.endswith(".gguf")]
        for f in consumed:
            if f not in produced and f not in USER_WRITTEN and not f.startswith("<"):
                problems.append(f"{action} consumes {f} before any command produces it")
        produced.update(Path(d[k]).name for k in PRODUCING if d.get(k))
    assert not problems, "\n".join(problems)


def test_every_dot_slash_script_gets_chmod():
    missing = [s for s in set(re.findall(r"\./(\S+\.py)", GUIDE)) if f"chmod +x {s}" not in GUIDE]
    assert not missing, missing


def test_quoted_log_lines_exist_in_code():
    missing = []
    for span in set(re.findall(r"`(\[[a-z]+\] [^`]+)`", GUIDE)):
        words = span.split()
        core = " ".join(words[:2]).split("{")[0]
        if core not in CODE:
            missing.append(span)
    assert not missing, missing
