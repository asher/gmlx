#!/usr/bin/env python3
"""Run the built-in clients of ``gmlx launch`` in a real Apple container.

The script writes a user config in a scratch HOME, starts its own server on
a free port with an official model that calls tools, and runs each group of
checks through ``gmlx launch``. Every launch gets the scratch HOME and XDG
folders in its environment, so the real ``~/.config/gmlx`` is never read or
written. The groups:

- doctor: ``gmlx doctor`` reports the container row and a private home, and
  the dry runs of claude-code, opencode and open-webui print the
  ``container run`` command, write the configuration into the private home,
  name the web port, record nothing and start nothing.
- clients: each built-in client completes one real turn against the model.
  claude-code, opencode, pi, omp, hermes, goose and dsh (headless profile)
  write a file into the shared project folder through a tool call, which the
  script reads on the Mac. aichat answers with a word that its prompt does
  not hold, and elia does so in its terminal interface in a pty. Open WebUI
  and the dsh web app start with ``--detach``, answer at ``http://[::1]``,
  show in ``--list`` and stop with ``--stop``, which frees the port. Open
  WebUI also answers one chat through its own API.
- images: ``packages`` adds a package and its removal rebuilds the image
  with the reason; a ``build`` Containerfile from a client's ``:base`` runs,
  and a changed file in its context rebuilds it with a line that names the
  file; ``image`` with ``command: image`` pulls and runs a public image, and
  the next launch deletes the images of the old ``build``; ``--image`` runs
  another image for one launch; ``--rebuild`` builds without the cache.
- home: the private home lasts from one launch to the next and differs
  between projects, no host config file reaches the guest, and
  ``--remove-home`` asks, removes the home and names the volume it keeps.
- seeds: a seed is copied, copied again after a change on the Mac, kept
  with the ``--reseed`` line after a change on both sides, replaced by
  ``--reseed``, and a credential path is refused.
- shares: the project is shared read-write at its path, ``--mount :ro`` is
  read-only, ``--no-mount-cwd`` shares nothing, the home folder and a
  credential folder are refused, and a commit works in a linked worktree.
- network: ``network: none`` blocks the internet while the model answers,
  ``forward`` reaches a server on the Mac's 127.0.0.1, and ``volumes`` keep
  their data from one launch to the next.
- ssh: ``ssh_agent`` with a throwaway agent lists only the throwaway key in
  the guest, and with the setting off the guest has no agent.
- sessions: a second launch joins a running session, ``--shell`` joins it
  too, ``--list`` and ``gmlx status`` show it, ``--stop`` ends it, a Ctrl-C
  and a SIGTERM end a session with the codes the docs give, and a launch
  killed with SIGKILL leaves a container that ``--list`` names and that the
  next launch in the project removes.
- media: from inside the guest, requests through the session socket with
  oversized images, a body over the session limit and a FLAC of three hours
  of silence get a 4xx, the server stays up and its resident memory stays
  within a bound.
- leftovers: always runs last. After the sessions end, no container of the
  run may remain. The script then deletes the run's volumes and images,
  restores each image reference it moved, removes the scratch folder, and
  checks that nothing of the run is left: containers, volumes, images,
  ``container`` processes, the builder, and files in the real launch
  folders of the Mac user.

    python tests/e2e/run_launch_container_e2e.py
    python tests/e2e/run_launch_container_e2e.py --only clients --clients pi,aichat

It needs Apple container 1.5.0 or newer with its service running, the guest
entry from ``scripts/build_guest_entry.py``, and an official model of the
tools role under the models root. It prints SKIP and exits 0 when one of
them is missing. The first build of an image needs network access. The
server never uses port 8091 or 8092. Exit status 0 means every check passed.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import glob
import http.server
import json
import os
import pwd
import re
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from container_e2e import (  # noqa: E402
    Background, Check, PeakRss, boxes, builder_state, container, container_ready,
    expect_plain, fetch, free_port, get_json, get_with_host, image_refs, plain, refused,
    rss_bytes, stop_group, volume_names, wait_until,
)
from models import ModelRegistry  # noqa: E402
from pty_session import PtyProcess  # noqa: E402
from server_proc import ServerProc  # noqa: E402

CLIENTS = ("claude-code", "opencode", "pi", "omp", "hermes", "goose", "aichat", "elia",
           "open-webui", "dsh")
GROUPS = ("doctor", "clients", "images", "home", "seeds", "shares", "network", "ssh",
          "sessions", "media", "leftovers")
RESERVED_PORTS = {8091, 8092}
WEB_PORTS = range(3100, 3200)
MODEL_ID = "e2e-model"
BUSYBOX = "docker.io/library/busybox:1.37.0"
WHISPER_GLOB = "~/.cache/huggingface/hub/models--mlx-community--whisper-*/snapshots/*/"
# A tool turn asks the client to write this file with this content.
FILE_PROMPT = ("Create the file {name} in the current folder with exactly this content: "
               "{marker}. Then reply DONE.")
# A chat turn asks for a word that the prompt does not hold, so the echo of
# the prompt cannot pass the check.
WORD_PROMPT = ("Write the word {a} and then the word ZULU with no space between them, and "
               "nothing else.")
ANSWERS_AT = r"runs in the background(?: for \S+)? at (http://\[::1\]:(\d+)/\S*?)\.?$"
LONG_PROMPT = "Write a 3000 word essay about rivers, with many details."


def shipped_versions(repo: str) -> dict[str, str]:
    """The VERSION of each stage of the shipped Containerfile."""
    text = open(os.path.join(repo, "gmlx", "container", "files", "Containerfile")).read()
    out, stage = {}, None
    for line in text.splitlines():
        m = re.match(r"FROM \S+ AS (\S+)", line)
        if m:
            stage = m[1]
        m = re.match(r"ARG VERSION=(\S+)", line)
        if m and stage:
            out[stage] = m[1]
    return out


def solid_png(width: int, height: int, rgb: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """A PNG of one color. It is small on disk and large once decoded."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    row = b"\x00" + bytes(rgb) * width
    z = zlib.compressobj(9)
    parts = [z.compress(row) for _ in range(height)]
    parts.append(z.flush())
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", b"".join(parts)) + chunk(b"IEND", b""))


def silence_flac(path: str, seconds: int) -> str | None:
    """Write a FLAC of silence at 16 kHz mono with ffmpeg or flac. The
    reason when neither is installed or the encode fails, else None."""
    if shutil.which("ffmpeg"):
        done = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                               "-i", "anullsrc=r=16000:cl=mono", "-t", str(seconds), "-c:a",
                               "flac", "-y", path], capture_output=True, text=True)
        return None if done.returncode == 0 else f"ffmpeg failed: {done.stderr.strip()}"
    if shutil.which("flac"):
        raw = path + ".raw"
        with open(raw, "wb") as f:
            block = b"\x00" * 32000
            for _ in range(seconds):
                f.write(block)
        done = subprocess.run(["flac", "--silent", "--force-raw-format", "--endian=little",
                               "--sign=signed", "--channels=1", "--bps=16",
                               "--sample-rate=16000", "-f", "-o", path, raw],
                              capture_output=True, text=True)
        os.unlink(raw)
        return None if done.returncode == 0 else f"flac failed: {done.stderr.strip()}"
    return "neither ffmpeg nor flac is installed"


class MarkerServer:
    """A small HTTP server on the Mac's 127.0.0.1 that answers every GET
    with a marker and counts the requests."""

    def __init__(self, port: int, marker: str):
        self.hits = 0
        outer = self

        class Page(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits += 1
                body = marker.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Page)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Run:
    """What the groups share: the options, the scratch folders, the config
    writer, the log, the checks, the processes the run started and what
    must be cleaned up at the end."""

    def __init__(self, a, repo: str, root: str, port: int, model: str, stt: str | None,
                 log: str, check: Check):
        self.a, self.repo, self.root, self.port = a, repo, root, port
        self.model, self.stt, self.log, self.check = model, stt, log, check
        self.prefix = "e2e" + secrets.token_hex(3)
        self.home = os.path.join(root, "home")
        self.data = os.path.join(root, "data")
        self.cache = os.path.join(root, "cache")
        self.box = os.path.join(root, "box")
        for folder in (self.home, os.path.join(self.home, ".config", "gmlx"), self.data,
                       self.cache, self.box, os.path.join(root, "t"),
                       os.path.join(root, "work"), os.path.join(root, "state")):
            os.makedirs(folder, exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("GMLX_", "OPENAI_", "ANTHROPIC_", "SSH_", "XDG_", "GIT_"))
               and k not in ("CLAUDE_CONFIG_DIR", "HERMES_HOME", "DSH_HOME", "ZDOTDIR",
                             "GNUPGHOME", "GH_CONFIG_DIR", "CARGO_HOME", "VIRTUAL_ENV")}
        env.update({"HOME": self.home, "XDG_DATA_HOME": self.data,
                    "XDG_CACHE_HOME": self.cache,
                    "XDG_CONFIG_HOME": os.path.join(self.home, ".config"),
                    "XDG_STATE_HOME": os.path.join(root, "state"),
                    "TMPDIR": os.path.join(root, "t"),
                    "PYTHONPATH": repo})
        self.env = env
        self.config_path = os.path.join(self.home, ".config", "gmlx", "gmlx.yaml")
        self.server_config = os.path.join(root, "server.yaml")
        self.pids: set[int] = set()
        self.sessions: list[Background] = []
        self.detached: list[tuple[str, str]] = []
        self.versions = shipped_versions(repo)
        self.cleanups: list[tuple[str, object]] = []
        self.server: ServerProc | None = None
        self.plain = plain_tags(a.python, repo)
        self.write_server_config()
        self.write_config()

    # Folders and config

    def work(self, name: str) -> str:
        """A project folder of this run. Its name starts with the run's
        prefix, so the project key that launch gives it does too."""
        folder = os.path.join(self.root, "work", f"{self.prefix}-{name}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def write_server_config(self) -> None:
        server = {"host": "127.0.0.1", "port": self.port, "menubar": False, "no_auth": True,
                  "cache": {"enabled": True}, "defaults": {"model": MODEL_ID}}
        if self.stt:
            server["stt"] = self.stt
        entry = {"path": self.model, "pin": True}
        if not self.a.thinking:
            entry["overrides"] = {"sampling": {"enable_thinking": False}}
        doc = {"server": server, "models": {MODEL_ID: entry}}
        with open(self.server_config, "w") as f:
            json.dump(doc, f, indent=1)

    def write_config(self, container: dict | None = None, clients: dict | None = None) -> None:
        """The user config of the scratch HOME, JSON being YAML: container
        mode on, no browser, goose in its one-shot form, and the group's own
        container settings."""
        merged = {"goose": {"command": ["goose"]}}
        for name, cfg in (clients or {}).items():
            merged[name] = {**merged.get(name, {}), **cfg}
        block = {"enabled": True, "open_browser": False, **(container or {}),
                 "clients": merged}
        doc = {"server": {"host": "127.0.0.1", "port": self.port, "menubar": False},
               "launch": {"container": block}}
        with open(self.config_path, "w") as f:
            json.dump(doc, f, indent=1)
        with open(self.log, "a") as f:
            f.write(f"\n# config: {json.dumps(block)}\n")

    # Commands

    def _argv(self, *args: str) -> list[str]:
        return [self.a.python, "-P", "-m", "gmlx", *args]

    def gmlx(self, *args: str, cwd: str, timeout: float, env: dict | None = None
             ) -> tuple[int, str]:
        """Run ``gmlx ARGS`` in ``cwd`` with no terminal, in a session of its
        own, and return its exit code, or -1 after a timeout, and its output."""
        argv = self._argv(*args)
        t0 = time.monotonic()
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)}\n")
        proc = subprocess.Popen(argv, cwd=cwd, env={**self.env, **(env or {})},
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.pids.add(proc.pid)
        shown = f"gmlx {' '.join(args)}"
        if len(shown) > 160:
            shown = shown[:157] + "..."
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            out = stop_group(proc)
            with open(self.log, "a") as f:
                f.write(out + f"\n# timed out after {timeout:.0f}s\n")
            print(f"  {shown} -> timed out after {timeout:.0f}s", flush=True)
            return -1, out
        except BaseException:
            stop_group(proc)
            raise
        out = out or ""
        with open(self.log, "a") as f:
            f.write(out + f"\n# exit {proc.returncode} in {time.monotonic() - t0:.0f}s\n")
        print(f"  {shown} -> exit {proc.returncode} in {time.monotonic() - t0:.0f}s", flush=True)
        return proc.returncode, out

    def launch(self, client: str, *args: str, cwd: str, timeout: float | None = None,
               env: dict | None = None) -> tuple[int, str]:
        head = args[:args.index("--")] if "--" in args else args
        if "--detach" in head:
            self.detached.append((client, cwd))
        return self.gmlx("launch", client, *args, cwd=cwd,
                         timeout=timeout or self.a.launch_timeout, env=env)

    def shell(self, client: str, script: str, *flags: str, cwd: str,
              timeout: float | None = None, env: dict | None = None) -> tuple[int, str]:
        """Run ``script`` with ``sh -c`` semantics in the client's image, in a
        shell session that the launch starts."""
        return self.launch(client, *flags, "--shell", "--", "-c", script, cwd=cwd,
                           timeout=timeout, env=env)

    def background(self, client: str, *args: str, cwd: str, env: dict | None = None
                   ) -> Background:
        argv = self._argv("launch", client, *args)
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)} (in the background)\n")
        session = Background(argv, cwd, {**self.env, **(env or {})}, self.log)
        self.pids.add(session.proc.pid)
        self.sessions.append(session)
        return session

    @contextlib.contextmanager
    def pty(self, *args: str, cwd: str):
        argv = self._argv(*args)
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)} (pty)\n")
            with PtyProcess(argv, env=dict(self.env), log=f, cwd=cwd) as p:
                self.pids.add(p.proc.pid)
                # Reap launch as a shell does: --stop waits until the launch
                # it signals is gone, and a zombie still counts as there.
                threading.Thread(target=p.proc.wait, daemon=True).start()
                yield p

    def stop_sessions(self) -> None:
        for session in self.sessions:
            session.stop()
        self.sessions.clear()
        for client, cwd in self.detached:
            self.gmlx("launch", client, "--stop", cwd=cwd, timeout=120)
        self.detached.clear()

    # What the run made

    def mine(self, box) -> bool:
        return box.project.startswith(self.prefix) or (box.pid.isdigit()
                                                       and int(box.pid) in self.pids)

    def running(self, client: str | None = None, cwd: str | None = None) -> list:
        """The running containers of the run, of ``client`` when given, and
        of the project of ``cwd`` when given."""
        key = os.path.basename(cwd) if cwd else None
        out = []
        for b in boxes(all_states=False):
            if not self.mine(b) or b.state != "running" or not b.client:
                continue
            if client and b.client != client:
                continue
            if key and not b.project.startswith(key + "-"):
                continue
            out.append(b)
        return out

    def home_of(self, client: str, cwd: str | None) -> str | None:
        """The private home of ``client`` for the project of ``cwd``, or of
        the default project."""
        base = os.path.join(self.data, "gmlx", "launch", client, "projects")
        if cwd is None:
            path = os.path.join(base, "default", "home")
            return path if os.path.isdir(path) else None
        found = glob.glob(os.path.join(base, glob.escape(os.path.basename(cwd)) + "-*", "home"))
        return found[0] if found else None

    def server_log(self) -> str:
        try:
            with open(os.path.join(self.a.out, "server.log"), errors="replace") as f:
                return f.read()
        except OSError:
            return ""


def file_turn(run: Run, client: str, cwd: str, args_for, *, timeout: float) -> None:
    """One tool turn of ``client`` that writes a file in the shared project
    folder, checked on the Mac."""
    name = f"e2e-{client}.txt"
    marker = f"E2E_{client.upper().replace('-', '_')}_OK"
    target = os.path.join(cwd, name)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(target)
    prompt = FILE_PROMPT.format(name=name, marker=marker)
    rc, out = run.launch(client, *args_for(prompt), cwd=cwd, timeout=timeout)
    try:
        with open(target) as f:
            content = f.read().strip()
    except OSError:
        content = None
    run.check(f"{client} writes a file in the shared folder through a tool call",
              rc == 0 and content == marker, f"exit {rc}, file {content!r}")
    image_line(run, client, out)
    run.check(f"{client}'s session ends with its container",
              wait_until(lambda: not run.running(client, cwd), 60))


def image_line(run: Run, client: str, out: str) -> None:
    version = run.versions.get(client)
    m = re.search(rf"\[launch\] image gmlx\.invalid/launch-{re.escape(client)}:[0-9a-f]+ with "
                  rf"(\S+) (\S+), (built|pulled) ", out)
    run.check(f"{client} runs the shipped image at the pinned version {version}",
              bool(m) and m[2] == version, m[0] if m else "no image line")


def group_doctor(run: Run) -> None:
    proj = run.work("dry")
    rc, out = run.launch("claude-code", "--config-only", cwd=proj, timeout=300)
    run.check("the dry run of claude-code prints the container run command and starts nothing",
              rc == 0 and "[launch] container dry run" in out and "container run --rm" in out
              and "-e IS_SANDBOX=1" in out and "--label gmlx.launch.client=claude-code" in out
              and not run.running("claude-code"), f"exit {rc}")
    run.check("the dry run names the image, the share and the session sockets",
              bool(re.search(r"\[launch\] image gmlx\.invalid/launch-claude-code:\w+: ", out))
              and f"[launch] sharing {proj} (read-write, working folder)" in out
              and "offers session sockets" in out, f"exit {rc}")
    run.check("the dry run gives Claude Code the model's context window",
              "CLAUDE_CODE_MAX_CONTEXT_TOKENS=" in out and "-e CLAUDE_CODE_MAX_CONTEXT_TOKENS "
              in out, f"exit {rc}")

    rc, out = run.launch("opencode", "--config-only", cwd=proj, timeout=300)
    home = run.home_of("opencode", proj)
    written = os.path.join(home or "", ".config", "gmlx", "opencode.json")
    try:
        with open(written) as f:
            text = f.read()
    except OSError:
        text = ""
    run.check("the dry run of opencode writes its configuration into the private home",
              rc == 0 and MODEL_ID in text and f"127.0.0.1:{run.port}" in text
              and not run.running("opencode"), f"exit {rc}, {written}")

    ports_file = os.path.join(run.data, "gmlx", "launch", "web-ports.json")
    before = open(ports_file).read() if os.path.exists(ports_file) else ""
    rc, out = run.launch("open-webui", "--config-only", cwd=proj, timeout=300)
    after = open(ports_file).read() if os.path.exists(ports_file) else ""
    # The command maps the guest's web socket to the port, which is the
    # same port on the Mac's [::1].
    named = re.search(r"gmlx-web\.sock=(\d+)", out)
    run.check("the dry run of open-webui names a Mac port from 3100 to 3199 and records none",
              rc == 0 and named is not None and int(named[1]) in WEB_PORTS and before == after
              and not run.running("open-webui"), f"exit {rc}, port {named and named[1]}")

    rc, out = run.gmlx("doctor", cwd=proj, timeout=300)
    run.check("gmlx doctor reports Apple container",
              bool(re.search(r"PASS\s+container\s+container 1\.\d+", out)), f"exit {rc}")
    run.check("gmlx doctor lists the private home with its project folder",
              bool(re.search(rf"home\s+claude-code: {re.escape(proj)}, ", out)), f"exit {rc}")


def group_clients(run: Run) -> None:
    chosen = run.a.clients.split(",") if run.a.clients else list(CLIENTS)
    proj = run.work("clients")
    t = run.a.turn_timeout
    terminal = {
        "claude-code": lambda p: ["--", "-p", p, "--dangerously-skip-permissions"],
        "opencode": lambda p: ["--", "run", p],
        "pi": lambda p: ["--", "-p", p],
        "omp": lambda p: ["--", "-p", p],
        "hermes": lambda p: ["--", "chat", "-q", p, "--yolo"],
        "goose": lambda p: ["--", "run", "-t", p],
    }
    for client in chosen:
        if client in terminal:
            file_turn(run, client, proj, terminal[client], timeout=t)
        elif client == "aichat":
            rc, out = run.launch("aichat", "--", WORD_PROMPT.format(a="AICHAT"), cwd=proj,
                                 timeout=t)
            run.check("aichat answers one prompt from the model", rc == 0
                      and "AICHATZULU" in out, f"exit {rc}")
            image_line(run, "aichat", out)
        elif client == "elia":
            client_elia(run, proj)
        elif client == "dsh":
            file_turn(run, "dsh", proj, lambda p: ["--dsh-profile", "headless", "--", p],
                      timeout=t)
            client_dsh_web(run)
        elif client == "open-webui":
            client_open_webui(run)
        else:
            run.check(f"{client} is a client this script knows", False)


def client_elia(run: Run, proj: str) -> None:
    with run.pty("launch", "elia", cwd=proj) as p:
        ui = expect_plain(p, "Enter your message", run.a.first_timeout)
        p.send(WORD_PROMPT.format(a="ELIA"))
        typed = expect_plain(p, "nothing else.", 30)
        p.send("\n")                                   # ctrl+j sends the message
        answered = expect_plain(p, "ELIAZULU", run.a.turn_timeout)
        p.send("\x1b")                                 # back to the home screen
        expect_plain(p, "Welcome to Elia", 20, count=2)
        p.send("\x1b")                                 # esc quits from there
        rc = p.wait_exit(60)
        text = plain(p.transcript)
    run.check("elia shows its interface in the container", ui and typed)
    run.check("elia answers one message from the model", answered)
    run.check("elia quits with esc and the launch exits 0", rc == 0, f"exit {rc}")
    m = re.search(r"image gmlx\.invalid/launch-elia:[0-9a-f]+ with (\S+) (\S+), ", text)
    run.check(f"elia runs the shipped image at the pinned version {run.versions.get('elia')}",
              bool(m) and m[2] == run.versions.get("elia"), m[0] if m else "no image line")
    run.check("elia's session ends with its container",
              wait_until(lambda: not run.running("elia"), 60))


def _detached_url(out: str) -> tuple[str | None, int | None]:
    for line in out.splitlines():
        m = re.search(ANSWERS_AT, line.strip())
        if m:
            return m[1], int(m[2])
    return None, None


def _list_and_stop(run: Run, client: str, cwd: str, port: int) -> None:
    rc, text = run.gmlx("launch", "--list", cwd=cwd, timeout=60)
    run.check(f"--list shows the detached {client} session with its address",
              rc == 0 and bool(re.search(rf"^{client}\s+.*\brunning\s+detached\s+"
                                         rf"http://\[::1\]:{port}/", text, re.M)), f"exit {rc}")
    rc, text = run.gmlx("status", cwd=cwd, timeout=60)
    run.check(f"gmlx status names the {client} session",
              f"launch session {client}" in text and f"http://[::1]:{port}/" in text,
              f"exit {rc}")
    rc, text = run.launch(client, "--stop", cwd=cwd, timeout=120)
    gone = wait_until(lambda: not run.running(client), 60)
    run.check(f"--stop ends the {client} session and frees its port",
              rc == 0 and f"[launch] stopped the {client} session" in text and gone
              and wait_until(lambda: refused("::1", port), 30), f"exit {rc}, gone {gone}")
    run.detached = [d for d in run.detached if d[0] != client]


def client_dsh_web(run: Run) -> None:
    web = run.work("dsh-web")
    rc, out = run.launch("dsh", "--detach", cwd=web, timeout=run.a.first_timeout)
    url, port = _detached_url(out)
    run.check("--detach of dsh returns once the web app runs at [::1] on a port from 3100 "
              "to 3199", rc == 0 and port in WEB_PORTS and "token=" in (url or ""),
              f"exit {rc}, {url}")
    if port is None:
        return
    status, _, headers = fetch("GET", url)
    cookie = (headers.get("Set-Cookie") or headers.get("set-cookie") or "").split(";")[0]
    status2, body, _ = fetch("GET", f"http://[::1]:{port}/", headers={"Cookie": cookie})
    run.check("the dsh address with its token signs in and the page answers",
              status == 303 and bool(cookie) and status2 == 200
              and b"DeepSeek Harness" in body, f"status {status}, then {status2}")
    status, _, _ = fetch("GET", f"http://[::1]:{port}/")
    run.check("the dsh page refuses a request without the token", status == 401,
              f"status {status}")
    status, body = get_with_host(port, f"localhost:{port}")
    run.check("another host name gets the 421 page",
              status == 421 and f"http://[::1]:{port}/" in body, f"status {status}")
    run.check("the Mac does not serve the app at 127.0.0.1", refused("127.0.0.1", port))
    _list_and_stop(run, "dsh", web, port)


def client_open_webui(run: Run) -> None:
    proj = run.work("owui")
    rc, out = run.launch("open-webui", "--detach", cwd=proj, timeout=run.a.first_timeout)
    url, port = _detached_url(out)
    run.check("--detach of open-webui returns once the app answers at [::1] on a port from "
              "3100 to 3199", rc == 0 and port in WEB_PORTS, f"exit {rc}, {url}")
    image_line(run, "open-webui", out)
    if port is None:
        return
    base = f"http://[::1]:{port}"
    status, body, _ = fetch("GET", base + "/")
    run.check("the Open WebUI page answers", status == 200 and b"Open WebUI" in body,
              f"status {status}")
    status, body, _ = fetch("POST", base + "/api/v1/auths/signup",
                           body=json.dumps({"name": "e2e", "email": "e2e@example.invalid",
                                            "password": "e2e-" + secrets.token_hex(8)}).encode(),
                           headers={"content-type": "application/json"})
    try:
        token = json.loads(body).get("token")
    except (ValueError, AttributeError):
        token = None
    run.check("the first account signs up through the app's API", status == 200 and bool(token),
              f"status {status}")
    if token:
        auth = {"Authorization": f"Bearer {token}", "content-type": "application/json"}
        status, body, _ = fetch("GET", base + "/api/models", headers=auth)
        run.check("Open WebUI lists the served model", status == 200
                  and MODEL_ID.encode() in body, f"status {status}")
        status, body, _ = fetch(
            "POST", base + "/api/chat/completions", headers=auth, timeout=run.a.turn_timeout,
            body=json.dumps({"model": MODEL_ID, "stream": False, "messages": [
                {"role": "user", "content": WORD_PROMPT.format(a="OWUI")}]}).encode())
        run.check("Open WebUI answers one chat through the session socket",
                  status == 200 and b"OWUIZULU" in body, f"status {status}, {body[:200]!r}")
    status, _ = get_with_host(port, f"localhost:{port}")
    run.check("Open WebUI refuses another host name with 421", status == 421, f"status {status}")
    _list_and_stop(run, "open-webui", proj, port)


def group_images(run: Run) -> None:
    proj = run.work("images")
    client = "aichat"
    run.write_config(clients={client: {"packages": ["jq"]}})
    rc, out = run.shell(client, "jq --version", cwd=proj, timeout=run.a.first_timeout)
    run.check("packages builds the image again with the package in it",
              rc == 0 and "building the aichat image" in out and "jq-1." in out, f"exit {rc}")

    run.write_config(clients={client: {"packages": ["jq", "tree"]}})
    rc, out = run.shell(client, "jq --version; tree --version", cwd=proj,
                        timeout=run.a.first_timeout)
    run.check("a changed packages list rebuilds the image and names the reason",
              rc == 0 and "jq-1." in out and "tree v" in out
              and "[launch] rebuilding because the packages list changed" in out, f"exit {rc}")

    run.write_config()
    with_pkgs = re.search(r"\[launch\] image (gmlx\.invalid/launch-aichat:[0-9a-f]{16}) with",
                          out)
    rc, out = run.shell(client, "command -v jq || echo NOJQ_$((1+1))", cwd=proj,
                        timeout=run.a.first_timeout)
    now = re.search(r"\[launch\] image (gmlx\.invalid/launch-aichat:[0-9a-f]{16}) with", out)
    run.check("removing the packages gives the image without them again",
              rc == 0 and "NOJQ_2" in out and now is not None and with_pkgs is not None
              and now[1] != with_pkgs[1], f"exit {rc}, {now and now[1]}")

    ctx = os.path.join(run.box, "aichat-build")
    os.makedirs(ctx, exist_ok=True)
    with open(os.path.join(ctx, "Containerfile"), "w") as f:
        f.write("FROM gmlx.invalid/launch-aichat:base\nCOPY data.txt /opt/e2e-data.txt\n")
    with open(os.path.join(ctx, "data.txt"), "w") as f:
        f.write("BUILD_ONE\n")
    run.write_config(clients={client: {"build": ctx}})
    rc, out = run.shell(client, "cat /opt/e2e-data.txt", cwd=proj, timeout=run.a.first_timeout)
    run.check("a build: Containerfile from the client's :base runs",
              rc == 0 and "BUILD_ONE" in out and "image gmlx.invalid/launch-aichat-build:" in out,
              f"exit {rc}")
    with open(os.path.join(ctx, "data.txt"), "w") as f:
        f.write("BUILD_TWO_CHANGED\n")
    rc, out = run.shell(client, "cat /opt/e2e-data.txt", cwd=proj, timeout=run.a.first_timeout)
    run.check("a changed file in the build context rebuilds the image, and the line names it",
              rc == 0 and "BUILD_TWO_CHANGED" in out
              and bool(re.search(r"\[launch\] rebuilding because \S*data\.txt changed", out)),
              f"exit {rc}")
    rc, out = run.launch(client, "--", "--version", cwd=proj, timeout=600)
    run.check("the client itself runs in the build: image", rc == 0
              and f"aichat {run.versions['aichat']}" in out, f"exit {rc}")

    run.write_config(clients={client: {"image": BUSYBOX, "command": "image"}})
    rc, out = run.launch(client, "--", "sh", "-c", "echo IMAGE_$((1+1)); uname -m", cwd=proj,
                         timeout=run.a.first_timeout)
    run.check("image: with command: image pulls and runs a public arm64 image",
              rc == 0 and "IMAGE_2" in out and "aarch64" in out
              and f"image {BUSYBOX}" in out, f"exit {rc}")
    left = [n for n in image_refs() if n.startswith("gmlx.invalid/launch-aichat-build")]
    run.check("the launch after build: was removed deletes the images of that build", not left,
              ", ".join(left))

    run.write_config()
    rc, out = run.shell(client, "echo ONCE_$((1+1)); test -x /usr/local/bin/aichat || echo "
                        "NO_AICHAT", "--image", BUSYBOX, cwd=proj, timeout=600)
    run.check("--image runs another image for one launch", rc == 0 and "ONCE_2" in out
              and "NO_AICHAT" in out, f"exit {rc}")

    ctx2 = os.path.join(run.box, "aichat-rebuild")
    os.makedirs(ctx2, exist_ok=True)
    with open(os.path.join(ctx2, "Containerfile"), "w") as f:
        f.write(f"FROM {BUSYBOX}\nRUN cat /proc/sys/kernel/random/uuid > /e2e-stamp\n")
    run.write_config(clients={client: {"build": ctx2}})
    stamps = []
    for flags in ((), (), ("--rebuild",)):
        rc, out = run.shell(client, "echo STAMP=$(cat /e2e-stamp)", *flags, cwd=proj,
                            timeout=run.a.first_timeout)
        m = re.search(r"STAMP=([0-9a-f-]{36})", out)
        stamps.append((rc, m[1] if m else None, "building " in out))
    run.check("a second launch reuses the build", stamps[1][0] == 0
              and stamps[1][1] == stamps[0][1] and not stamps[1][2], repr(stamps[:2]))
    run.check("--rebuild builds the image again without the cache",
              stamps[2][0] == 0 and stamps[2][2] and stamps[2][1] not in (None, stamps[0][1]),
              repr(stamps))
    run.write_config()


def group_home(run: Run) -> None:
    a, b = run.work("home-a"), run.work("home-b")
    vol = f"{run.prefix}-hvol"
    run.write_config(clients={"claude-code": {"volumes": [f"{vol}:/e2e-hvol:1G"]}})
    secret = "HOSTSECRET_" + run.prefix
    os.makedirs(os.path.join(run.home, ".claude"), exist_ok=True)
    os.makedirs(os.path.join(run.home, ".ssh"), mode=0o700, exist_ok=True)
    with open(os.path.join(run.home, ".claude", "settings.json"), "w") as f:
        json.dump({"e2e": secret}, f)
    with open(os.path.join(run.home, ".claude.json"), "w") as f:
        json.dump({"e2e": secret}, f)
    with open(os.path.join(run.home, ".ssh", "id_e2e"), "w") as f:
        f.write(secret + "\n")

    rc, out = run.shell("claude-code", 'echo "$HOME" > /dev/null; echo PERSIST_A > '
                        '"$HOME/e2e-persist.txt"; echo HOME_IS=$HOME', cwd=a)
    m = re.search(r"HOME_IS=(\S+)", out)
    home_a = m[1] if m else None
    rc2, out2 = run.shell("claude-code", 'cat "$HOME/e2e-persist.txt"', cwd=a)
    run.check("the private home keeps a file from one launch to the next",
              rc == 0 and rc2 == 0 and "PERSIST_A" in out2, f"exits {rc}, {rc2}")
    mac = run.home_of("claude-code", a)
    run.check("the private home is the folder under the launch data on the Mac",
              mac is not None and home_a == mac
              and os.path.isfile(os.path.join(mac, "e2e-persist.txt")), f"{home_a} vs {mac}")
    rc, out = run.shell("claude-code", 'echo HOME_IS=$HOME; cat "$HOME/e2e-persist.txt" '
                        '2>/dev/null || echo NOT_HERE_$((1+1))', cwd=b)
    m = re.search(r"HOME_IS=(\S+)", out)
    run.check("another project gets another private home",
              rc == 0 and "NOT_HERE_2" in out and m is not None and m[1] != home_a, f"exit {rc}")

    real_home = pwd.getpwuid(os.getuid()).pw_dir
    probe = "; ".join(f'test -e "{p}" && echo "LEAK {p}"' for p in (
        os.path.join(run.home, ".claude"), os.path.join(run.home, ".claude.json"),
        os.path.join(run.home, ".ssh"), os.path.join(run.home, ".config", "gmlx"), real_home))
    rc, out = run.shell("claude-code", f'{probe}; grep -rIl {secret} "$HOME" /root /etc '
                        f'2>/dev/null; echo PROBE_DONE_$((1+1))', cwd=a)
    leaks = [line for line in out.splitlines() if line.startswith("LEAK ") or
             (secret in line and not line.startswith("#"))]
    run.check("no host config file and no host folder appears in the guest",
              rc == 0 and "PROBE_DONE_2" in out and not leaks, "; ".join(leaks))

    rc, out = run.launch("claude-code", "--remove-home", cwd=a, timeout=60)
    run.check("--remove-home with no terminal asks nothing and removes nothing",
              rc == 1 and "there is no terminal to ask on" in out
              and run.home_of("claude-code", a) is not None, f"exit {rc}")
    with run.pty("launch", "claude-code", "--remove-home", cwd=a) as p:
        asked = p.expect("? [y/N]", 60)
        if asked:
            p.sendline("y")
        rc = p.wait_exit(120)
        text = plain(p.transcript)
    m = re.search(rf"container volume delete ({re.escape(vol)}-[0-9a-f]{{8}})", text)
    run.check("--remove-home asks, removes the private home and its project folder",
              asked and rc == 0 and "[launch] removed " in text
              and run.home_of("claude-code", a) is None, f"exit {rc}")
    run.check("--remove-home names the project volume it keeps and the command that deletes it",
              m is not None and m[1] in volume_names(), m[0] if m else "no volume line")
    if m:
        done = container("volume", "delete", m[1])
        run.check("the volume that --remove-home named deletes with its command",
                  done.returncode == 0 and m[1] not in volume_names(), done.stderr.strip())
    run.write_config()


def group_seeds(run: Run) -> None:
    proj = run.work("seeds")
    notes = os.path.join(run.home, "e2e-notes")
    os.makedirs(notes, exist_ok=True)
    src = os.path.join(notes, "notes.md")
    shown = "~/e2e-notes/notes.md"
    read = 'cat "$HOME/e2e-notes/notes.md"'

    def write(text: str) -> None:
        with open(src, "w") as f:
            f.write(text + "\n")

    write("SEED_ONE")
    run.write_config(clients={"claude-code": {"seed": [shown]}})
    rc, out = run.shell("claude-code", read, cwd=proj)
    run.check("a seed is copied into the private home and named",
              rc == 0 and "SEED_ONE" in out and f"[launch] seed: copied {shown}" in out,
              f"exit {rc}")
    write("SEED_TWO_LONGER")
    rc, out = run.shell("claude-code", read, cwd=proj)
    run.check("a seed changed on the Mac is copied again",
              rc == 0 and "SEED_TWO_LONGER" in out
              and f"[launch] seed: copied {shown} again, because it changed on the Mac" in out,
              f"exit {rc}")
    rc, _ = run.shell("claude-code", 'echo GUEST_EDIT > "$HOME/e2e-notes/notes.md"', cwd=proj)
    write("SEED_THREE_LONGEST")
    rc2, out = run.shell("claude-code", read, cwd=proj)
    run.check("a seed changed on both sides keeps the guest copy and names --reseed",
              rc == 0 and rc2 == 0 and "GUEST_EDIT" in out
              and f"[launch] seed: {shown} changed on the Mac and in the private home" in out
              and "--reseed replaces it" in out, f"exits {rc}, {rc2}")
    rc, out = run.shell("claude-code", read, "--reseed", cwd=proj)
    run.check("--reseed replaces the copy with the Mac file",
              rc == 0 and "SEED_THREE_LONGEST" in out and f"[launch] seed: copied {shown}" in out,
              f"exit {rc}")

    os.makedirs(os.path.join(run.home, ".ssh"), mode=0o700, exist_ok=True)
    with open(os.path.join(run.home, ".ssh", "id_e2e_seed"), "w") as f:
        f.write("not a key\n")
    run.write_config(clients={"claude-code": {"seed": ["~/.ssh/id_e2e_seed"]}})
    rc, out = run.shell("claude-code", "echo SHOULD_NOT_RUN_$((1+1))", cwd=proj)
    run.check("a seed in a credential folder is refused",
              rc not in (0, -1) and "seed: will not copy ~/.ssh/id_e2e_seed, because" in out
              and "SHOULD_NOT_RUN_2" not in out, f"exit {rc}")
    run.write_config()


def group_shares(run: Run) -> None:
    proj, other = run.work("shares"), run.work("shares-ro")
    with open(os.path.join(other, "ro.txt"), "w") as f:
        f.write("RO_CONTENT\n")
    rc, out = run.shell("claude-code", "pwd; echo GUEST_WROTE > rw.txt; echo PWD_DONE", cwd=proj)
    try:
        wrote = open(os.path.join(proj, "rw.txt")).read().strip()
    except OSError:
        wrote = None
    run.check("the project is shared read-write at its own path",
              rc == 0 and f"\n{proj}\n" in f"\n{out}" and wrote == "GUEST_WROTE"
              and f"[launch] sharing {proj} (read-write, working folder)" in out, f"exit {rc}")

    rc, out = run.shell("claude-code", f'cat "{other}/ro.txt"; touch "{other}/new.txt" '
                        '&& echo WROTE_RO || echo RO_REFUSED_$((1+1))',
                        "--mount", f"{other}:ro", cwd=proj)
    run.check("--mount PATH:ro shares a folder read-only",
              rc == 0 and "RO_CONTENT" in out and "RO_REFUSED_2" in out
              and not os.path.exists(os.path.join(other, "new.txt"))
              and f"[launch] sharing {other} (read-only)" in out, f"exit {rc}")
    rc, out = run.shell("claude-code", "touch ro-cwd.txt && echo WROTE || echo "
                        "CWD_RO_$((1+1))", "--mount", ".:ro", cwd=proj)
    run.check("--mount .:ro shares the project read-only",
              rc == 0 and "CWD_RO_2" in out and not os.path.exists(os.path.join(proj, "ro-cwd.txt")),
              f"exit {rc}")

    rc, out = run.shell("claude-code", f'test -e "{proj}" && echo VISIBLE || echo '
                        'HIDDEN_$((1+1))', "--no-mount-cwd", cwd=proj)
    run.check("--no-mount-cwd shares nothing and says so",
              rc == 0 and "HIDDEN_2" in out and "the current folder is not shared" in out,
              f"exit {rc}")

    rc, out = run.shell("claude-code", "echo SHOULD_NOT_RUN", cwd=run.home, timeout=120)
    run.check("the home folder is refused with its message",
              rc == 1 and "will not share the current folder ~, because it is your home folder. "
              "Launch from a project folder, or pass --no-mount-cwd." in out, f"exit {rc}")
    ssh = os.path.join(run.home, ".ssh")
    os.makedirs(ssh, mode=0o700, exist_ok=True)
    rc, out = run.shell("claude-code", "echo SHOULD_NOT_RUN", cwd=ssh, timeout=120)
    run.check("a credential folder is refused with its message",
              rc == 1 and "will not share the current folder ~/.ssh, because" in out
              and "SHOULD_NOT_RUN" not in out, f"exit {rc}")

    repo, wt = run.work("repo"), os.path.join(run.root, "work", f"{run.prefix}-wt")
    genv = {**run.env, "GIT_CONFIG_NOSYSTEM": "1"}
    git = shutil.which("git", path="/opt/homebrew/bin:/usr/bin") or "git"

    def g(*args: str, cwd: str = repo) -> subprocess.CompletedProcess:
        return subprocess.run([git, *args], cwd=cwd, env=genv, capture_output=True, text=True)

    g("config", "--global", "user.name", "E2E Tester")
    g("config", "--global", "user.email", "e2e@example.invalid")
    g("init", "-q", "-b", "main")
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("one\n")
    g("add", "a.txt")
    g("commit", "-q", "-m", "e2e base")
    made = g("worktree", "add", "-q", "-b", "e2e-wt", wt)
    rc, out = run.shell("claude-code", "echo two > b.txt && git add b.txt && git commit -q -m "
                        "e2e-guest-commit && echo COMMITTED_$((1+1))", cwd=wt)
    log = g("log", "--all", "--format=%s %an").stdout
    run.check("a commit in a linked worktree works through the shared git folder",
              made.returncode == 0 and rc == 0 and "COMMITTED_2" in out
              and "e2e-guest-commit E2E Tester" in log
              and bool(re.search(r"\[launch\] sharing \S+/\.git\S* \(read-write, ", out)),
              f"exit {rc}, worktree {made.returncode} {made.stderr.strip()}")


def group_network(run: Run) -> None:
    proj = run.work("network")
    curl_net = ("curl -sS -m 20 -o /dev/null -w 'NET_CODE=%{http_code}\\n' https://example.com "
                "2>&1; echo NET_RC=$?")
    ask = WORD_PROMPT.format(a="NETTEST")
    body = json.dumps({"model": MODEL_ID, "max_tokens": 64,
                       "messages": [{"role": "user", "content": ask}]})
    model = (f"curl -sS -m 600 -H 'content-type: application/json' -d '{body}' "
             f"http://127.0.0.1:{run.port}/v1/chat/completions")
    rc, out = run.shell("claude-code", curl_net, cwd=proj)
    online = "NET_CODE=200" in out
    print(f"  the guest {'reaches' if online else 'does not reach'} https://example.com "
          "with the default network", flush=True)
    run.write_config(container={"network": "none"})
    rc, out = run.shell("claude-code", f"{curl_net}; {model}", cwd=proj)
    run.check("network: none blocks the internet in the guest",
              rc == 0 and "NET_RC=0" not in out and "NET_CODE=200" not in out
              and "with network none, the client reaches only the gmlx server" in out,
              f"exit {rc}, internet reachable without it: {online}")
    run.check("under network: none the model still answers through the session socket",
              "NETTESTZULU" in out, f"exit {rc}")

    marker = f"FWD_{run.prefix}"
    fport = free_port(RESERVED_PORTS | set(WEB_PORTS) | {run.port})
    server = MarkerServer(fport, marker)
    run.cleanups.append(("stop the forward test server", server.close))
    run.write_config(container={"network": "none", "forward": [fport]})
    rc, out = run.shell("claude-code", f"curl -sS -m 30 http://127.0.0.1:{fport}/; echo", cwd=proj)
    run.check("forward reaches a loopback service on the Mac, also under network: none",
              rc == 0 and marker in out and server.hits >= 1
              and f"[launch] forwarding the container's 127.0.0.1:{fport} to Mac port {fport}"
              in out, f"exit {rc}, {server.hits} requests on the Mac")

    vol, gvol = f"{run.prefix}-nvol", f"{run.prefix}-gvol"
    run.write_config(container={"volumes": [f"{gvol}:/e2e-gvol:1G"]},
                     clients={"claude-code": {"volumes": [f"{vol}:/e2e-vol:1G"]}})
    rc, out = run.shell("claude-code", "mkdir -p /e2e-vol/sub /e2e-gvol/sub && echo VOLDATA > "
                        "/e2e-vol/sub/f && echo GVOLDATA > /e2e-gvol/sub/f && echo WROTE_$((1+1))",
                        cwd=proj)
    rc2, out2 = run.shell("claude-code", "cat /e2e-vol/sub/f /e2e-gvol/sub/f", cwd=proj)
    names = volume_names()
    per_project = [n for n in names if re.fullmatch(rf"{re.escape(vol)}-[0-9a-f]{{8}}", n)]
    run.check("volumes keep their data from one launch to the next",
              rc == 0 and "WROTE_2" in out and rc2 == 0 and "VOLDATA" in out2
              and "GVOLDATA" in out2, f"exits {rc}, {rc2}")
    run.check("a client volume gets the project's name and a global one keeps its name",
              len(per_project) == 1 and gvol in names, f"{per_project}, {gvol in names}")
    run.write_config()


def group_ssh(run: Run) -> None:
    proj = run.work("ssh")
    folder = os.path.join(run.root, "agent")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    sock = os.path.join(folder, "s.sock")
    key = os.path.join(folder, "id_e2e")
    comment = f"e2e-throwaway-{run.prefix}"
    started = subprocess.run(["ssh-agent", "-a", sock, "-s"], capture_output=True, text=True,
                             env={**run.env})
    m = re.search(r"SSH_AGENT_PID=(\d+)", started.stdout)
    if started.returncode != 0 or not m:
        run.check("a throwaway ssh-agent starts", False, started.stderr.strip())
        return
    agent_pid = int(m[1])

    def stop_agent() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(agent_pid, signal.SIGTERM)
        wait_until(lambda: not _alive(agent_pid), 10)

    run.cleanups.append(("stop the throwaway ssh-agent", stop_agent))
    aenv = {**run.env, "SSH_AUTH_SOCK": sock}
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", key],
                   env=aenv, capture_output=True, check=True)
    added = subprocess.run(["ssh-add", key], env=aenv, capture_output=True, text=True)
    fingerprint = subprocess.run(["ssh-keygen", "-lf", key + ".pub"], capture_output=True,
                                 text=True).stdout.split()[1:2]
    run.check("the throwaway key is in the throwaway agent", added.returncode == 0,
              added.stderr.strip())

    run.write_config(container={"ssh_agent": True})
    rc, out = run.shell("claude-code", "ssh-add -l; echo AGENT_RC=$?", cwd=proj,
                        env={"SSH_AUTH_SOCK": sock})
    listed = [line for line in out.splitlines() if re.match(r"^\d+ SHA256:", line)]
    run.check("ssh_agent gives the guest the throwaway agent with only its key",
              rc == 0 and "AGENT_RC=0" in out and len(listed) == 1 and comment in listed[0]
              and bool(fingerprint) and fingerprint[0] in listed[0], "; ".join(listed))
    run.write_config()
    rc, out = run.shell("claude-code", "ssh-add -l; echo AGENT_RC=$?", cwd=proj,
                        env={"SSH_AUTH_SOCK": sock})
    run.check("without ssh_agent the guest has no agent",
              rc == 0 and "AGENT_RC=2" in out and comment not in out, f"exit {rc}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _requests(run: Run) -> int:
    return run.server_log().count("Generation queued")


def group_sessions(run: Run) -> None:
    proj = run.work("sessions")
    version = run.versions["claude-code"]
    with run.pty("launch", "claude-code", "--shell", cwd=proj) as first:
        up = first.expect("# ", 300)
        rc, out = run.launch("claude-code", "--", "--version", cwd=proj, timeout=300)
        count = len(run.running("claude-code", proj))
        run.check("a second launch joins the running session",
                  up and rc == 0 and "joining the running claude-code session" in out
                  and f"{version} (Claude Code)" in out and count == 1,
                  f"exit {rc}, {count} containers")
        rc, out = run.shell("claude-code", "echo JOINED_$((1+1))", cwd=proj, timeout=300)
        count = len(run.running("claude-code", proj))
        run.check("--shell joins the running session",
                  rc == 0 and "opening a shell in the running claude-code session" in out
                  and "JOINED_2" in out and count == 1, f"exit {rc}, {count} containers")
        rc, out = run.gmlx("launch", "--list", cwd=proj, timeout=60)
        run.check("--list shows the session",
                  rc == 0 and bool(re.search(rf"^claude-code\s+{re.escape(proj)}\s+running\s",
                                             out, re.M)), f"exit {rc}")
        rc, out = run.gmlx("status", cwd=proj, timeout=60)
        run.check("gmlx status shows the session",
                  f"launch session claude-code for {proj}: running" in out, f"exit {rc}")
        rc, out = run.launch("claude-code", "--stop", cwd=proj, timeout=120)
        ended = first.wait_exit(90)
    gone = wait_until(lambda: not run.running("claude-code", proj), 60)
    run.check("--stop ends the session and its container",
              rc == 0 and f"[launch] stopped the claude-code session for {proj}." in out
              and ended is not None and gone, f"exit {rc}, first launch exit {ended}")

    script = "trap 'echo TRAPPED_INT; exit 42' INT; echo READY_$((1+1)); sleep 600 & wait"
    with run.pty("launch", "claude-code", "--shell", "--", "-c", script, cwd=proj) as p:
        ready = p.expect("READY_2", 300)
        p.send("\x03")
        rc = p.wait_exit(60)
        trapped = "TRAPPED_INT" in p.transcript
    run.check("a Ctrl-C reaches the program in the container, and launch exits with its code",
              ready and trapped and rc == 42, f"exit {rc}")
    run.check("the container is gone after the Ctrl-C",
              wait_until(lambda: not run.running("claude-code", proj), 60))

    base = _requests(run)
    with run.pty("launch", "claude-code", "--", "-p", LONG_PROMPT, cwd=proj) as p:
        seen = wait_until(lambda: _requests(run) > base or p.proc.poll() is not None, 300, 0.5)
        p._drain(0.2)
        t0 = time.monotonic()
        p.send("\x03")
        rc = p.wait_exit(60)
        took = time.monotonic() - t0
    run.check("a Ctrl-C ends claude-code during a turn, and launch exits with claude's code",
              seen and rc in (0, 130) and took < 30, f"exit {rc} after {took:.1f}s")
    run.check("the container is gone after claude-code's Ctrl-C",
              wait_until(lambda: not run.running("claude-code", proj), 60))

    base = _requests(run)
    session = run.background("claude-code", "--", "-p", LONG_PROMPT, cwd=proj)
    seen = wait_until(lambda: _requests(run) > base or session.proc.poll() is not None, 300,
                      0.5)
    t0 = time.monotonic()
    session.signal(signal.SIGTERM)
    rc = session.wait(60)
    took = time.monotonic() - t0
    run.check("SIGTERM to launch ends claude-code's session with 143",
              seen and rc == 143 and took < 30, f"exit {rc} after {took:.1f}s")
    run.check("the container is gone after the SIGTERM",
              wait_until(lambda: not run.running("claude-code", proj), 60))

    session = run.background("claude-code", "--", "-p", LONG_PROMPT, cwd=proj)
    up = wait_until(lambda: bool(run.running("claude-code", proj))
                    or session.proc.poll() is not None, 300, 0.5)
    names = [b.name for b in run.running("claude-code", proj)]
    session.signal(signal.SIGKILL)
    session.wait(30)
    rc, out = run.gmlx("launch", "--list", cwd=proj, timeout=60)
    named = bool(names) and all(f"container stop {n}" in out for n in names)
    run.check("after a SIGKILL of launch, --list names the leftover container with its stop "
              "command", up and rc == 0 and named, f"exit {rc}, containers {names}")
    rc, out = run.shell("claude-code", "echo AFTER_$((1+1))", cwd=proj, timeout=300)
    cleaned = wait_until(lambda: not any(b.name in names for b in boxes()), 90)
    run.check("the next launch in the project runs and removes the leftover container",
              rc == 0 and "AFTER_2" in out and cleaned, f"exit {rc}, removed {cleaned}")
    stray = wait_until(lambda: not _processes(names), 60)
    run.check("no container process of the killed launch is left", stray,
              "; ".join(_processes(names)))


def _processes(words: list[str]) -> list[str]:
    """The command lines of running processes that name one of ``words``."""
    if not words:
        return []
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines()
            if any(w in line for w in words) and "ps -axo" not in line]


def group_media(run: Run) -> None:
    proj = run.work("media")
    sp = run.server

    def body(parts: list[dict]) -> bytes:
        return json.dumps({"model": MODEL_ID, "max_tokens": 8, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Describe this."}, *parts]}]}
        ).encode()

    def image(png: bytes) -> dict:
        return {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}

    big = solid_png(8192, 8192)
    payloads = {
        "many-large-pngs": body([image(big)] * 16),
        "one-huge-png": body([image(solid_png(9000, 9000))]),
        "too-many-pngs": body([image(solid_png(8, 8))] * 65),
        "over-session-body": b'{"model":"' + MODEL_ID.encode() + b'","messages":[{"role":'
                             b'"user","content":"' + b"a" * (33 << 20) + b'"}]}',
    }
    flac_long = os.path.join(proj, "silence-3h.flac")
    why = silence_flac(flac_long, 3 * 3600)
    flac_short = os.path.join(proj, "silence-1s.flac")
    if why is None:
        why = silence_flac(flac_short, 1)
    if why is None:
        audio = base64.b64encode(open(flac_long, "rb").read()).decode()
        payloads["flac-in-chat"] = body([{"type": "input_audio",
                                          "input_audio": {"data": audio, "format": "flac"}}])
    else:
        print(f"SKIP: the FLAC checks, because {why}.", flush=True)
    for name, data in payloads.items():
        with open(os.path.join(proj, name + ".json"), "wb") as f:
            f.write(data)

    url = f"http://127.0.0.1:{run.port}/v1"
    lines = []
    for name in payloads:
        lines.append(f"curl -sS -m 600 -o {name}.out -w 'RESULT {name} %{{http_code}}\\n' "
                     f"-H 'content-type: application/json' --data-binary @{name}.json "
                     f"{url}/chat/completions")
    if why is None and run.stt:
        lines.insert(0, f"curl -sS -m 600 -o warm.out -w 'RESULT warm-stt %{{http_code}}\\n' "
                        f"-F file=@silence-1s.flac {url}/audio/transcriptions")
        lines.append(f"curl -sS -m 600 -o flac-stt.out -w 'RESULT flac-stt %{{http_code}}\\n' "
                     f"-F file=@silence-3h.flac {url}/audio/transcriptions")
    elif not run.stt:
        print("SKIP: the transcription check, because no local whisper model was found "
              f"({WHISPER_GLOB}).", flush=True)

    pid = sp.proc.pid if sp and sp.proc else None
    if pid is None:
        run.check("the server runs", False)
        return
    if why is None and run.stt:
        rc, out = run.shell("claude-code", lines[0], cwd=proj, timeout=900)
        run.check("a short FLAC is transcribed through the session socket",
                  "RESULT warm-stt 200" in out, f"exit {rc}")
        lines = lines[1:]
    baseline = rss_bytes(pid) or 0
    with PeakRss(pid) as peak:
        rc, out = run.shell("claude-code", "; ".join(lines), cwd=proj, timeout=1800)
    results = dict(re.findall(r"^RESULT (\S+) (\d+)$", out, re.M))

    def said(name: str) -> str:
        try:
            with open(os.path.join(proj, name + ".out"), errors="replace") as f:
                return f.read()[:400]
        except OSError:
            return ""

    expect = {"many-large-pngs": ("400", "pixels"), "one-huge-png": ("400", "pixels"),
              "too-many-pngs": ("4", ""), "over-session-body": ("413", "32 MiB"),
              "flac-in-chat": ("4", ""), "flac-stt": ("400", "samples")}
    for name, (code, words) in expect.items():
        if name not in payloads and not (name == "flac-stt" and why is None and run.stt):
            continue
        got = results.get(name, "")
        run.check(f"{name} from the guest is refused with a 4xx",
                  got.startswith(code) and got.startswith("4") and words in said(name),
                  f"status {got or 'none'}: {said(name)[:240]}")
    health = get_json(f"http://127.0.0.1:{run.port}/health")
    run.check("the server stays up after the refusals",
              health is not None and _alive(pid) and sp.proc.poll() is None, str(health))
    grew = peak.peak - baseline
    run.check(f"the server's resident memory grows by less than {run.a.media_mb} MiB",
              grew < run.a.media_mb << 20,
              f"baseline {baseline >> 20} MiB, peak {peak.peak >> 20} MiB")


GROUP_RUNNERS = {"doctor": group_doctor, "clients": group_clients, "images": group_images,
                 "home": group_home, "seeds": group_seeds, "shares": group_shares,
                 "network": group_network, "ssh": group_ssh, "sessions": group_sessions,
                 "media": group_media}


def _snapshot_real_launch() -> dict[str, tuple]:
    """The files of the Mac user's real launch folders, with size and time,
    which the run must not touch."""
    real = pwd.getpwuid(os.getuid()).pw_dir
    out = {}
    for top in (os.path.join(real, ".local", "share", "gmlx", "launch"),
                os.path.join(real, ".cache", "gmlx", "launch")):
        for folder, dirs, files in os.walk(top):
            for name in files + dirs:
                path = os.path.join(folder, name)
                with contextlib.suppress(OSError):
                    st = os.lstat(path)
                    out[path] = (st.st_size, st.st_mtime_ns)
    return out


def plain_tags(python: str, repo: str) -> set[str]:
    """The tag of each client's image from the shipped recipe with no
    packages, as the checkout computes it."""
    code = ("from gmlx.container import images\n"
            f"for c in {(*CLIENTS, 'runtime-python')!r}: print(images.shipped_tag(c, []))")
    done = subprocess.run([python, "-P", "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": repo}, cwd="/")
    return set(done.stdout.split())


def _kept(refs: dict[str, str], plain: set[str]) -> set[str]:
    """The references that --keep-images keeps: the plain shipped tags, the
    :base tags, and the digest references of the plain images."""
    digests = {refs[t] for t in plain if t in refs}
    repos = {t.split(":", 1)[0] for t in plain}
    return {n for n, d in refs.items() if n in plain
            or (n.split(":", 1)[0] in repos and n.endswith(":base"))
            or (n.split("@", 1)[0] in repos and "@" in n and d in digests)}


def _guarded(what: str, step, check: Check) -> None:
    try:
        step()
    except KeyboardInterrupt:
        check(f"cleanup: {what}", False, "skipped after Ctrl-C")
    except Exception as e:                                   # noqa: BLE001
        check(f"cleanup: {what}", False, f"{type(e).__name__}: {e}")


def clean_up(run: Run, before: dict) -> None:
    """Stop what the run started, check that launch left no container,
    delete the run's volumes and images, put back each image reference the
    run moved, remove the scratch folder, and check that nothing is left."""
    check = run.check
    check.group = "leftovers"
    _guarded("stop the sessions", run.stop_sessions, check)

    def containers() -> None:
        left = [b for b in boxes() if run.mine(b) and b.client]
        check("no container of the run is left once its sessions end", not left,
              ", ".join(map(repr, left)))
        for b in left:
            container("stop", "--time", "10", b.name)
            container("delete", "--force", b.name)
        check("the leftover containers are removed",
              not [b for b in boxes() if run.mine(b) and b.client])

    _guarded("remove the leftover containers", containers, check)
    _guarded("stop the server", lambda: run.server and run.server.stop(), check)
    for what, step in run.cleanups:
        _guarded(what, step, check)

    def volumes() -> None:
        new = sorted(volume_names() - before["volumes"])
        ours = [v for v in new if v.startswith(run.prefix)]
        other = [v for v in new if not v.startswith(run.prefix)]
        check("the run made no volume other than the ones its settings name", not other,
              ", ".join(other))
        for name in ours:
            container("volume", "delete", name)
        check("the run's volumes are deleted",
              not [v for v in volume_names() if v.startswith(run.prefix)])

    _guarded("delete the volumes", volumes, check)

    def images() -> None:
        # With --keep-images the plain shipped client images stay, so a later
        # run reuses them. Every other image of the run goes.
        now = image_refs()
        keep = _kept(now, run.plain) if run.a.keep_images else set()
        old = {n: d for n, d in before["images"].items() if n not in keep}
        by_digest: dict[str, list[str]] = {}
        for name, digest in now.items():
            by_digest.setdefault(digest, []).append(name)
        for name, digest in old.items():
            if now.get(name) == digest:
                continue
            source = next(iter(by_digest.get(digest, [])), None)
            if source:
                container("image", "tag", source, name)
        added = sorted(set(image_refs()) - set(old) - keep)
        if added:
            container("image", "delete", *added)
        after = {n: d for n, d in image_refs().items() if n not in keep}
        changed = sorted(n for n in set(old) | set(after) if old.get(n) != after.get(n))
        check("the image store holds the same references as before the run"
              + (", apart from the plain shipped client images" if keep else ""), not changed,
              ", ".join(changed[:12]) + (f" and {len(changed) - 12} more"
                                         if len(changed) > 12 else ""))

    _guarded("delete the images", images, check)

    def builder() -> None:
        state = builder_state()
        check("the image builder is not left running",
              state != "running" or before["builder"] == "running", f"state {state}")

    _guarded("check the builder", builder, check)

    def processes() -> None:
        stray = [p for p in _processes([run.prefix, run.root])
                 if "run_launch_container_e2e" not in p]
        check("no process of the run is left", not stray, "; ".join(stray[:5]))

    _guarded("check the processes", processes, check)

    def real_home() -> None:
        now = _snapshot_real_launch()
        changed = sorted(p for p in set(now) | set(before["real"])
                         if now.get(p) != before["real"].get(p))
        check("the real launch folders of the Mac user are unchanged", not changed,
              ", ".join(changed[:5]))

    _guarded("check the real launch folders", real_home, check)

    def scratch() -> None:
        homes = glob.glob(os.path.join(run.data, "gmlx", "launch", "*", "projects", "*", "home"))
        print(f"removing {len(homes)} private homes with the scratch folder", flush=True)
        if run.a.keep:
            print(f"kept {run.root}", flush=True)
            return
        shutil.rmtree(run.root, ignore_errors=True)
        check("the scratch folder with the private homes is removed",
              not os.path.exists(run.root))

    _guarded("remove the scratch folder", scratch, check)


def find_stt() -> str | None:
    found = sorted(glob.glob(os.path.expanduser(WHISPER_GLOB)))
    tiny = [f for f in found if "turbo-q4" in f] or [f for f in found if "turbo" in f] or found
    return tiny[0].rstrip("/") if tiny else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root)
    ap.add_argument("--model", help="a GGUF path, in place of the tools role of models.py")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that runs the server and gmlx")
    ap.add_argument("--out", help="the folder for the logs (default: a fresh temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder")
    ap.add_argument("--keep-images", action="store_true",
                    help="keep the images the run built or pulled, so a later run reuses them")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model think, which makes each turn slower")
    ap.add_argument("--stt", default="auto",
                    help="a speech-to-text model folder for the media group, or none "
                         f"(default: the first of {WHISPER_GLOB})")
    ap.add_argument("--clients", help="a comma-separated list of clients for the clients group")
    ap.add_argument("--first-timeout", type=float, default=1800.0,
                    help="seconds for a launch that builds or pulls an image")
    ap.add_argument("--launch-timeout", type=float, default=900.0,
                    help="seconds for any other launch")
    ap.add_argument("--turn-timeout", type=float, default=1800.0,
                    help="seconds for a launch that runs one model turn")
    ap.add_argument("--media-mb", type=int, default=1024,
                    help="the most the server's resident memory may grow in the media group")
    ap.add_argument("--only", action="append", choices=GROUPS, metavar="GROUP",
                    help=f"run only this group, repeatable: {', '.join(GROUPS)}. leftovers "
                         "always runs last")
    a = ap.parse_args()

    why = container_ready()
    if why:
        print(f"SKIP: {why}.")
        return 0
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    entry = os.path.join(repo, "gmlx", "container", "guest", "gmlx-entry")
    if not os.path.isfile(entry):
        print(f"SKIP: the guest entry {entry} is not built (python scripts/build_guest_entry.py).")
        return 0
    if a.model:
        model = os.path.expanduser(a.model)
    else:
        reg = ModelRegistry(root=a.models_root)
        paths = reg.role_paths("tools")
        if not paths:
            print(f"SKIP: no model of the tools role under {a.models_root} "
                  f"({reg.role_groups('tools')}).")
            return 0
        model = paths[0]
    if not os.path.exists(model):
        print(f"SKIP: {model} does not exist.")
        return 0
    stt = None if a.stt == "none" else (find_stt() if a.stt == "auto"
                                        else os.path.expanduser(a.stt))
    groups = [g for g in GROUPS if g != "leftovers" and (not a.only or g in a.only)]

    a.out = a.out or tempfile.mkdtemp(prefix="gmlx-container-e2e-")
    os.makedirs(a.out, exist_ok=True)
    # The real path, since launch refuses a build: path that goes through a
    # link, such as /tmp.
    root = os.path.realpath(tempfile.mkdtemp(prefix="gmlx-ce-", dir="/tmp"))
    port = free_port(RESERVED_PORTS | set(WEB_PORTS))
    check = Check()
    log = os.path.join(a.out, "launch.log")
    before = {"images": image_refs(), "volumes": volume_names(), "builder": builder_state(),
              "real": _snapshot_real_launch()}
    run = Run(a, repo, root, port, model, stt, log, check)
    print(f"model {model}\nstt {stt}\nserver port {port}\nscratch {root}\nprefix {run.prefix}\n"
          f"logs {a.out}\ngroups {', '.join(groups)}, leftovers", flush=True)

    shared = ("HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME",
              "TMPDIR", "PYTHONPATH")
    run.server = ServerProc(["--config", run.server_config],
                            log_path=os.path.join(a.out, "server.log"), port=port,
                            python=a.python, env_extra={k: run.env[k] for k in shared})
    try:
        _run_groups(run, groups)
    finally:
        print("\n== leftovers", flush=True)
        clean_up(run, before)

    print(f"\nlogs: {a.out}")
    if check.failed:
        print(f"FAILED {len(check.failed)} of {len(check.rows)}: " + "; ".join(check.failed))
        return 1
    print(f"ALL {len(check.rows)} CHECKS PASSED")
    return 0


def _run_groups(run: Run, groups: list[str]) -> None:
    """Start the server, then run each group. An error ends only its own
    group, which then counts as a failed check."""
    try:
        run.server.start()
        run.server.wait_ready(timeout=900)
    except Exception as e:                                   # noqa: BLE001
        run.check("the server starts", False, f"{type(e).__name__}: {e}")
        return
    print(f"server ready at {run.server.base_url}", flush=True)
    for name in groups:
        print(f"\n== {name}", flush=True)
        run.check.group = name
        t0 = time.monotonic()
        try:
            GROUP_RUNNERS[name](run)
        except Exception as e:                               # noqa: BLE001
            run.check("the group completes", False, f"{type(e).__name__}: {e}")
        finally:
            run.stop_sessions()
            run.write_config()
            print(f"== {name} took {time.monotonic() - t0:.0f}s", flush=True)
    run.check.group = ""


if __name__ == "__main__":
    sys.exit(main())
