#!/usr/bin/env python3
"""End-to-end smoke for the interactive ``gmlx chat`` terminal UI.

This is the one tier that drives the *real* program in a *real* pseudo-terminal -
the only way to exercise what only exists under a tty: prompt_toolkit's live editing
session, the streaming reply, and the termios Esc-cancel. The faster, deterministic
loop + session coverage lives in ``tests/test_chat_e2e.py`` (CPU, no tty, no model);
this fills the integration gap those mocks can't reach.

It spawns ``gmlx chat <gguf>`` over a pty (stdlib :mod:`pty`, no ``pexpect``) and
walks a scripted session:

  1. **load + banner** - the ready banner reports ``editor: prompt_toolkit`` (the
     real TUI engaged, not the line-editor shim); the one-line ``[load]`` summary
     is checked after turn 1, since the default background load defers it to the
     first-turn join;
  2. **two turns** - a reply streams to completion on each of two prompts over
     one persistent KV cache, and the stat line carries the v2 segments
     (``prompt ... tok @ ... gen ... tok @ ... ctx ...``);
  3. **slash commands, live** - ``/temp`` + ``/sampling``, the ``/model`` card,
     ``/stats`` totals, ``/theme`` + ``/render`` switches, ``/system`` and
     ``/thinking-budget``, a markdown-rendered turn, ``/copy`` (NOTE: touches the
     real clipboard when pbcopy/xclip is present), and ``/retry`` + ``/undo``
     over the turn ledger;
  4. **Esc-cancel** - a long reply is interrupted mid-stream with Esc and the REPL
     reports ``reply canceled`` and returns to the prompt (cbreak/termios path);
  5. **sessions + quit** - ``/save`` + ``/export`` + ``/sessions`` (into an
     isolated ``XDG_DATA_HOME``), then ``/exit`` ends the process cleanly;
  6. **resume** - a second launch with ``--resume`` prints the recap and streams
     a post-resume turn (the deferred full-history prefill).

An optional **multimodal** arm runs when a VLM GGUF + projector are present
(``gemma4_e2b`` + ``gemma4_e2b_mmproj``): it stages the bundled ``assets/cats.jpg``
with ``/image`` and generates a reply about it (the picture-says-"cat" check is
reported, not gated - it depends on the model).

Not ``test_``-prefixed, so pytest skips it - it loads a model and needs the GPU. Run
it directly with the project interpreter::

    python tests/e2e/run_chat_pty_e2e.py                 # text arm (+ vlm if present)
    python tests/e2e/run_chat_pty_e2e.py --no-vlm        # text arm only
    python tests/e2e/run_chat_pty_e2e.py --keep --out ./chat-pty-out
    python tests/e2e/run_chat_pty_e2e.py --print-pull    # how to fetch the model

A missing model (or missing console script) is a **SKIP** (exit 0), never a failure.
Also collected by pytest as ``tests/test_chat_pty.py`` when ``KQUANT_TEST_GGUF_DIR``
is set (the shared integration-test gate).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelRegistry           # noqa: E402
from pty_session import PtyProcess          # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
LOAD_TIMEOUT = 180.0                        # generous: cold model load + Metal warmup
REPLY_TIMEOUT = 90.0
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _child_env(xdg=None) -> dict:
    """Child env: offline, a real TERM, sessions isolated under ``xdg``, and this
    checkout first on PYTHONPATH so the console script tests the repo the harness
    lives in (not whatever install it happens to point at)."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TERM="xterm-256color", COLORTERM="truecolor")
    repo = str(Path(__file__).resolve().parents[2])
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo + (os.pathsep + prior if prior else "")
    if xdg is not None:
        env["XDG_DATA_HOME"] = str(xdg)
    return env


def _expect_any(chat, needles, timeout=10.0):
    """First of ``needles`` to appear past the cursor, or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seg = chat.transcript[chat._cursor:]
        for n in needles:
            if n in seg:
                chat._cursor = chat.transcript.find(n, chat._cursor) + len(n)
                return n
        chat._drain(0.3)
    return None


def _cli(python: str) -> str:
    """The ``gmlx`` console script next to the interpreter."""
    cand = os.path.join(os.path.dirname(python), "gmlx")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"no gmlx console script next to {python} - install the package")


class Check:
    """A named structural assertion + its evidence, for the report + exit code."""

    def __init__(self):
        self.rows = []          # (name, passed, gated, detail)

    def add(self, name, passed, detail="", *, gated=True):
        self.rows.append((name, bool(passed), gated, detail))
        flag = "PASS" if passed else ("FAIL" if gated else "warn")
        print(f"  [{flag}] {name}" + (f" - {detail}" if detail else ""))
        return passed

    def ok(self):
        return all(p for _n, p, gated, _d in self.rows if gated)


def _run_text_arm(cli: str, gguf: str, python: str, log_path: str,
                  checks: Check, xdg=None) -> None:
    export_md = Path(log_path).with_name("chat-export.md")
    argv = [cli, "chat", gguf, "--no-history", "--temp", "0.0",
            "--max-tokens", "256", "--seed", "0"]
    env = _child_env(xdg)
    print(f"[text] {' '.join(argv)}")
    with open(log_path, "w") as log, PtyProcess(argv, env=env, log=log) as chat:
        # 1. load + banner
        loaded = chat.expect("Esc cancels a reply", timeout=LOAD_TIMEOUT)
        checks.add("model loads + chat banner appears", loaded,
                   "" if loaded else f"no banner in {LOAD_TIMEOUT:.0f}s\n" + chat.tail())
        if not loaded:
            return
        checks.add("real prompt_toolkit TUI engaged (tty)",
                   "editor: prompt_toolkit" in chat.transcript,
                   "banner did not report prompt_toolkit" if
                   "editor: prompt_toolkit" not in chat.transcript else "")

        # 2. two turns over one persistent KV cache
        chat.sendline("Reply with exactly one word: ready")
        t1 = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("turn 1 streams a reply to completion", t1,
                   "" if t1 else "no stat line\n" + chat.tail())
        checks.add("quiet load prints the [load] summary line",
                   "[load] " in _plain(chat.transcript),
                   "no [load] line by the end of turn 1" if
                   "[load] " not in _plain(chat.transcript) else "")
        statv2 = re.search(r"prompt \S+ tok @ .+gen \S+ tok @ .+ctx \S+",
                           _plain(chat.transcript))
        checks.add("stat line v2 (prompt / gen / ctx segments)", bool(statv2),
                   "" if statv2 else _plain(chat.tail(400)))
        chat.sendline("Now reply with exactly one word: again")
        t2 = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("turn 2 streams a reply (multi-turn)", t2,
                   "" if t2 else "no second stat line\n" + chat.tail())

        # 3. a slash command, live
        chat.sendline("/temp 0.8")
        set_ok = chat.expect("temp = 0.8", timeout=10.0)
        chat.sendline("/sampling")
        show_ok = chat.expect("sampling:", timeout=10.0)
        checks.add("slash command adjusts + echoes sampling", set_ok and show_ok,
                   "/temp or /sampling produced no echo" if not (set_ok and show_ok) else "")

        # 3b. info surfaces + live theme/render switches + runtime knobs
        chat.sendline("/model")
        card = chat.expect("arch", timeout=10.0)
        chat.sendline("/stats")
        totals = chat.expect("turn", timeout=10.0)
        checks.add("/model card + /stats totals", card and totals,
                   "" if card and totals else _plain(chat.tail(400)))
        chat.sendline("/theme nord")
        th = chat.expect("theme = nord", timeout=10.0)
        chat.sendline("/render lite")
        rd = chat.expect("render = lite", timeout=10.0)
        checks.add("/theme + /render switch live", th and rd,
                   "" if th and rd else _plain(chat.tail(400)))
        chat.sendline("/system")
        sysshow = chat.expect("no system prompt set", timeout=10.0)
        chat.sendline("/thinking-budget 64")
        tb = chat.expect("thinking-budget = 64", timeout=10.0)
        checks.add("/system shows + /thinking-budget sets", sysshow and tb,
                   "" if sysshow and tb else _plain(chat.tail(400)))

        # 3c. a markdown-rendered turn (lite mode is active from 3b)
        chat.sendline("Show a markdown bullet list of three fruits.")
        md = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("markdown turn streams under the lite renderer", md,
                   "" if md else "no stat line\n" + chat.tail())
        checks.add("bullet glyph rendered (model-dependent)",
                   "\u2022" in _plain(chat.transcript), gated=False)

        # 3d. /copy - either clipboard notice proves the command plumbing (a
        # thinking model may leave no answer text; both are correct outputs).
        chat.sendline("/copy")
        cp = _expect_any(chat, ("chars (", "no answer text"), timeout=10.0)
        checks.add("/copy responds", cp is not None,
                   "" if cp else _plain(chat.tail(400)))

        # 3e. /retry + /undo over the turn ledger
        chat.sendline("/retry")
        rt = chat.expect("regenerating", timeout=10.0)
        rt2 = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("/retry regenerates the last reply", rt and rt2,
                   "" if rt and rt2 else _plain(chat.tail(400)))
        chat.sendline("/undo")
        un = chat.expect("last exchange removed", timeout=10.0)
        checks.add("/undo removes the last exchange", un,
                   "" if un else _plain(chat.tail(400)))

        # 4. Esc-cancel a reply mid-stream. Don't guess at timing (a fast GPU finishes
        # a short reply in well under a second): wait for prompt_toolkit to hand control
        # to generation - it disables bracketed paste ("\x1b[?2004l") as the prompt
        # session ends - so the Esc can't be swallowed by the line editor. The byte then
        # sits in the tty buffer until the cbreak drain catches it at the first streamed
        # chunk, cancelling at token 1 regardless of speed. A couple of nudges cover any
        # accept-timing race; the loop stops at the first cancel so no stray Esc reaches
        # the next prompt.
        chat.sendline("Count slowly from 1 to 100, one number per line.")
        chat.expect("\x1b[?2004l", timeout=15.0)
        canceled = False
        for _ in range(8):
            chat.send("\x1b")
            if chat.expect("reply canceled", timeout=0.5):
                canceled = True
                break
        checks.add("Esc cancels a streaming reply", canceled,
                   "" if canceled else "no cancel notice\n" + chat.tail())
        chat.sendline("")                   # flush any escape state

        # 5. sessions: save + export + list, then quit
        chat.sendline("/save pty-e2e")
        sv = chat.expect("saved", timeout=10.0)
        chat.sendline(f"/export {export_md}")
        ex = chat.expect("exported", timeout=10.0)
        chat.sendline("/sessions")
        ls = chat.expect("pty-e2e", timeout=10.0)
        checks.add("/save + /export + /sessions", sv and ex and ls,
                   "" if sv and ex and ls else _plain(chat.tail(600)))
        chat.sendline("/exit")
        code = chat.wait_exit(timeout=30.0)
        checks.add("'/exit' ends the process cleanly", code == 0,
                   f"exit code {code}")
    exported = export_md.is_file() and "## User" in export_md.read_text()
    checks.add("exported markdown transcript on disk", exported,
               "" if exported else str(export_md))


def _run_resume_arm(cli: str, gguf: str, log_path: str, checks: Check,
                    xdg=None) -> None:
    """Relaunch against the session `/save`d by the text arm: the recap prints
    and the next turn prefills the restored history (the deferred replay)."""
    argv = [cli, "chat", gguf, "--no-history", "--temp", "0.0",
            "--max-tokens", "64", "--seed", "0",
            "--resume", "pty-e2e", "--no-autosave"]
    env = _child_env(xdg)
    print(f"[resume] {' '.join(argv)}")
    with open(log_path, "w") as log, PtyProcess(argv, env=env, log=log) as chat:
        recap = chat.expect("resumed 'pty-e2e'", timeout=LOAD_TIMEOUT)
        checks.add("--resume prints the session recap", recap,
                   "" if recap else "no recap\n" + chat.tail())
        if not recap:
            return
        chat.sendline("Reply with exactly one word: back")
        turn = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("post-resume turn streams (history prefills)", turn,
                   "" if turn else "no stat line\n" + chat.tail())
        chat.sendline("/exit")
        code = chat.wait_exit(timeout=30.0)
        checks.add("resume arm exits cleanly", code == 0, f"exit code {code}")


def _run_vlm_arm(cli: str, gguf: str, mmproj: str, log_path: str, checks: Check) -> None:
    image = ASSETS / "cats.jpg"
    if not image.exists():
        checks.add("vlm: bundled test image present", False, str(image))
        return
    argv = [cli, "chat", gguf, "--mmproj", mmproj, "--no-history",
            "--temp", "0.0", "--max-tokens", "64", "--seed", "0",
            "--no-autosave"]
    env = _child_env()
    print(f"[vlm]  {' '.join(argv)}")
    with open(log_path, "w") as log, PtyProcess(argv, env=env, log=log) as chat:
        loaded = chat.expect("multimodal", timeout=LOAD_TIMEOUT)
        checks.add("vlm: loads + multimodal banner", loaded,
                   "" if loaded else "no multimodal banner\n" + chat.tail())
        if not loaded:
            return
        chat.sendline(f"/image {image}")
        staged = chat.expect("staged", timeout=15.0)
        checks.add("vlm: /image stages the picture", staged,
                   "" if staged else "no stage notice\n" + chat.tail())
        chat.sendline("What animal is in this image? Answer in one word.")
        replied = chat.expect("tok @", timeout=REPLY_TIMEOUT)
        checks.add("vlm: a reply streams with the image attached", replied,
                   "" if replied else "no reply\n" + chat.tail())
        # The model actually recognizing the cats is reported, not gated.
        if replied:
            window = chat.transcript[max(0, chat._cursor - 600):chat._cursor].lower()
            checks.add("vlm: reply mentions a cat (soft)", "cat" in window,
                       gated=False)
        chat.sendline("/exit")
        code = chat.wait_exit(timeout=30.0)
        checks.add("vlm: '/exit' exits cleanly", code == 0, f"exit code {code}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root,
                    help="root for the GGUF (default ~/llm/gguf)")
    ap.add_argument("--handle", default="qwen3_0_6b_q4",
                    help="registry handle for the text model (default qwen3_0_6b_q4)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip the multimodal arm even if a VLM GGUF is present")
    ap.add_argument("--print-pull", action="store_true",
                    help="print the pull command(s) for the needed model(s) and exit")
    ap.add_argument("--out", default=None, help="artifact dir (default: temp, removed)")
    ap.add_argument("--keep", action="store_true", help="keep logs + report on disk")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose console script + venv drives the child")
    a = ap.parse_args()

    reg = ModelRegistry(root=a.models_root)
    if a.print_pull:
        reg.print_bootstrap([a.handle, "gemma4_e2b", "gemma4_e2b_mmproj"])
        return 0

    try:
        cli = _cli(a.python)
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        return 0

    gguf = reg.find(a.handle)
    if gguf is None:
        print(f"SKIP: text model handle {a.handle!r} not found under {a.models_root}.")
        reg.print_bootstrap([a.handle])
        return 0

    tmp = a.out or tempfile.mkdtemp(prefix="gguf-chat-pty-e2e-")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    print(f"[e2e] cli={cli}\n[e2e] text model={gguf}\n[e2e] artifacts={tmp}")

    checks = Check()
    rc = 1
    try:
        print("=" * 64 + "\nTEXT ARM\n" + "=" * 64)
        xdg = os.path.join(tmp, "xdg")
        _run_text_arm(cli, gguf, a.python, os.path.join(tmp, "chat-text.log"),
                      checks, xdg=xdg)

        print("=" * 64 + "\nRESUME ARM\n" + "=" * 64)
        _run_resume_arm(cli, gguf, os.path.join(tmp, "chat-resume.log"),
                        checks, xdg=xdg)

        vlm, mmproj = reg.find("gemma4_e2b"), reg.find("gemma4_e2b_mmproj")
        if a.no_vlm:
            print("[vlm]  skipped (--no-vlm)")
        elif vlm and mmproj:
            print("=" * 64 + "\nMULTIMODAL ARM\n" + "=" * 64)
            _run_vlm_arm(cli, vlm, mmproj, os.path.join(tmp, "chat-vlm.log"), checks)
        else:
            print("[vlm]  skipped (no gemma4_e2b + mmproj on disk)")

        rc = _report(checks, Path(tmp))
    finally:
        if not a.keep and not a.out:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[e2e] artifacts kept under {tmp}")
    return rc


def _report(checks: Check, out_dir: Path) -> int:
    lines = ["# chat TUI pty e2e\n"]
    for name, passed, gated, detail in checks.rows:
        flag = "PASS" if passed else ("FAIL" if gated else "warn")
        lines.append(f"- [{flag}] {name}" + (f" - {detail.splitlines()[0]}" if detail else ""))
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report + "\n")
    print("\n" + "=" * 64)
    ok = checks.ok()
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
