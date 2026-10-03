#!/usr/bin/env python3
"""Run custom agents from launch.agents in a real Apple container.

The script writes a user config in a scratch HOME that defines five agents,
writes their projects, and launches them through ``gmlx launch`` against a
server it starts itself on a free port. The checks come in groups:

- runtime: a LangChain project installs its dependencies with uv on the
  first launch and completes one tool call through the session socket. The
  second launch downloads nothing, a launch under ``network: none`` starts
  from the synced volume, and the agent's exit code comes back.
- dry-run: the dry run of a web agent maps the guest's web socket to its
  ``web_port``, records no Mac port and starts no container.
- web: a web agent that is a PEP 723 script answers at ``http://[::1]`` on a
  Mac port from 3100 to 3199. The app sees ``HOST``, its ``web_port`` as
  ``PORT``, and the ``[::1]`` Host. Another host name gets the 421 page, and
  the Mac answers at no other address. A second launch names the running
  app, a second project runs at once on a port of its own, SIGTERM ends the
  sessions, and the project keeps its port. ``--shell`` names ``uv run``,
  a second launch names it again, and the app that the shell starts that
  way answers. ``--remove-home`` names the site data and releases the port.
- join: a second launch joins a session that ``--shell`` holds, ``uv run``
  in the shell uses the agent's environment, a second ``--shell`` joins,
  and the session ends when its shell exits.
- signals: a Ctrl-C on the terminal reaches the agent once and the launch
  ends with 130. SIGTERM to launch reaches the agent once and stops the
  container.
- detach: ``--detach`` of the web agent returns once the app answers, and
  the app goes on answering with its output in the output file.
  ``--list`` and ``gmlx status`` show the session, a second ``--detach``
  names the app, and ``--stop`` ends it and frees the port. ``--detach`` of
  the agent returns once its container runs, a second one is refused, and
  ``--stop`` reaches the agent once. ``--stop`` also ends a session that a
  launch without ``--detach`` runs, and ``--detach`` refuses pi and
  ``--shell``.
- source: an agent whose ``source`` is the LangChain project runs from
  another folder with the source read-only, cannot write to it, and a stale
  ``uv.lock`` stops it with uv's message.
- build: an agent with its own ``build`` from the runtime base runs a script
  from its read-only source with uv in that image.
- api: an ``api: anthropic`` agent gets a reply from the Messages route
  through its session socket.
- doctor: ``gmlx doctor`` names an agent's home by the agent name and the
  project folder.

At the end the script answers yes to ``--remove-home`` in a pty for each
project it used, which removes the homes and the dependency volumes, and it
deletes the images the run created.

    python tests/e2e/run_launch_agents_e2e.py
    python tests/e2e/run_launch_agents_e2e.py --only web --only signals

It needs Apple container 1.5.0 or newer with its service running, the guest
entry from ``scripts/build_guest_entry.py``, and an official model that calls
tools under the models root. It prints SKIP and exits 0 when one of them is
missing. The first install also needs network access, and without it the
runtime checks fail. The real
``~/.config/gmlx/gmlx.yaml`` is never read or edited, and the server never
uses port 8091 or 8092. Exit status 0 means every check passed.
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelRegistry  # noqa: E402
from pty_session import PtyProcess  # noqa: E402
from server_proc import ServerProc, free_port  # noqa: E402

AGENT = "e2e-agent"        # the LangChain project, from its own folder
WEB = "e2e-web"            # a web app in a PEP 723 script
SRC = "e2e-src"            # the LangChain project as a read-only source
BUILD = "e2e-build"        # an own build from the runtime base
API = "e2e-api"            # api: anthropic
AGENTS = (AGENT, WEB, SRC, BUILD, API)
RESERVED_PORTS = {8091, 8092}
WEB_PORTS = range(3100, 3200)
RUNTIME_REPO = "gmlx.invalid/launch-runtime-python"
GROUPS = ("runtime", "dry-run", "web", "join", "signals", "detach", "source", "build", "api",
          "doctor")
# A group that needs the synced environment or the lockfile of the runtime
# group runs that group first.
NEEDS_RUNTIME = {"join", "signals", "detach", "source", "doctor"}
ANSWERS_AT = r"the web app answers at http://\[::1\]:(\d+)/"
OUTPUT_AT = r"its output goes to (\S+)\. gmlx launch --list"

PYPROJECT = '''[project]
name = "e2e-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["langchain-openai", "langchain-core"]

[project.scripts]
e2e-agent = "e2e_agent:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/e2e_agent"]
'''

AGENT_CODE = '''"""The agent of the launch-agents end-to-end script: one tool call, or
with --wait, a wait that names each SIGINT and SIGTERM it gets."""
import os
import signal
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def on_signal(signum, frame):
    # A second copy of the signal during the pause prints a second line.
    print("E2E_SIGINT" if signum == signal.SIGINT else "E2E_SIGTERM", flush=True)
    time.sleep(1)
    sys.exit(128 + signum)


def main():
    if "--wait" in sys.argv:
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        print("E2E_WAITING", flush=True)
        while True:
            time.sleep(1)
    code = int(sys.argv[sys.argv.index("--exit") + 1]) if "--exit" in sys.argv else 0
    print("E2E_START python", sys.version.split()[0], "model", os.environ.get("GMLX_MODEL"),
          flush=True)
    llm = ChatOpenAI(model=os.environ["GMLX_MODEL"], temperature=0, max_tokens=512)
    reply = llm.bind_tools([add]).invoke([HumanMessage(
        "Use the add tool to add 2 and 3. Call the tool, and write nothing else.")])
    calls = [(c["name"], c["args"]) for c in reply.tool_calls]
    if len(calls) != 1 or calls[0][0] != "add":
        print("E2E_FAIL tool calls", calls, repr(reply.content), flush=True)
        sys.exit(1)
    print("E2E_TOOL_CALL", calls[0][0], calls[0][1].get("a"), calls[0][1].get("b"), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
'''

WEB_CODE = '''# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""The web agent of the launch-agents end-to-end script: one JSON page that
names the address the app listens on and the Host of the request."""
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"marker": "E2E_WEB", "host": self.headers.get("Host"),
                           "env_host": os.environ.get("HOST"),
                           "env_port": os.environ.get("PORT")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


signal.signal(signal.SIGTERM, lambda *args: sys.exit(143))
server = ThreadingHTTPServer((os.environ["HOST"], int(os.environ["PORT"])), Page)
print("E2E_WEB_LISTENING", os.environ["HOST"], os.environ["PORT"], flush=True)
server.serve_forever()
'''

BUILD_CODE = '''# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Print the file that the agent's own Containerfile adds."""
import sys

print("E2E_BUILD", open("/opt/e2e-marker").read().strip(), sys.version.split()[0], flush=True)
'''

API_CODE = '''# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One request to the Messages route with the Anthropic variables only."""
import json
import os
import urllib.request

body = json.dumps({"model": os.environ["ANTHROPIC_MODEL"], "max_tokens": 256,
                   "messages": [{"role": "user", "content": "Say OK."}]}).encode()
request = urllib.request.Request(
    os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages", data=body,
    headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
             "content-type": "application/json"})
with urllib.request.urlopen(request, timeout=300) as response:
    reply = json.load(response)
print("E2E_API", reply.get("type"), reply.get("role"), "key", os.environ["ANTHROPIC_API_KEY"],
      "openai", "OPENAI_API_KEY" in os.environ,
      "window", os.environ.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"), flush=True)
'''

CONTAINERFILE = f'''FROM {RUNTIME_REPO}:base
RUN printf 'E2E_BUILD_MARKER\\n' > /opt/e2e-marker
'''


class Check:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""),
              flush=True)
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def container(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["container", *args], capture_output=True, text=True, check=check)


def container_ready() -> str | None:
    """None when Apple container 1.5.0 or newer runs, else the reason."""
    if shutil.which("container") is None:
        return "Apple container is not installed"
    out = container("--version").stdout
    m = re.search(r"version (\d+)\.(\d+)\.(\d+)", out)
    if not m or tuple(int(x) for x in m.groups()) < (1, 5, 0):
        return f"Apple container 1.5.0 or newer is needed, found {out.strip() or 'none'}"
    status = container("system", "status")
    if status.returncode != 0 or not re.search(r"^status\s+running\s*$", status.stdout, re.M):
        return "the container service is not running (container system start)"
    return None


def _rows(*args: str) -> list[dict]:
    try:
        rows = json.loads(container(*args, "--format", "json").stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [r for r in rows if isinstance(r, dict)]


def image_names() -> set[str]:
    return {(r.get("configuration") or r).get("name", "") for r in _rows("image", "list")} - {""}


def volume_names() -> set[str]:
    return {(r.get("configuration") or r).get("name", "") for r in _rows("volume", "list")} - {""}


def _labelled(rows: list[dict], agents) -> list[str]:
    keys = {f"agent-{a}" for a in agents}
    names = []
    for row in rows:
        conf = row.get("configuration") or {}
        if (conf.get("labels") or {}).get("gmlx.launch.client") in keys:
            names.append(row.get("id") or conf.get("id", ""))
    return [n for n in names if n]


def running(agent: str) -> list[str]:
    """The running containers of ``agent``, by the label launch gives them."""
    return _labelled(_rows("ls"), [agent])


def wait_until(test, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if test():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def get_with_host(port: int, host: str) -> tuple[int | None, str]:
    """GET / at [::1] and ``port`` with the Host header ``host``."""
    conn = http.client.HTTPConnection("::1", port, timeout=10)
    try:
        conn.request("GET", "/", headers={"Host": host})
        r = conn.getresponse()
        return r.status, r.read().decode(errors="replace")
    except OSError as e:
        return None, repr(e)
    finally:
        conn.close()


def refused(host: str, port: int) -> bool:
    """Whether nothing on the Mac accepts a connection at ``host`` and ``port``."""
    try:
        with socket.create_connection((host, port), timeout=3):
            return False
    except OSError:
        return True


def write_scratch(root: str, port: int, guest_port: int, repo: str) -> dict:
    """The scratch HOME with the user config of the five agents, the launch
    state folders, and the agents' folders. PYTHONPATH names the checkout,
    so ``-m gmlx`` runs its code."""
    home = os.path.join(root, "home")
    work = {name: os.path.join(root, "work", name)
            for name in ("agent", "web", "web2", "elsewhere", "tools")}
    box = os.path.join(root, "box")
    for folder in (*work.values(), box, os.path.join(home, ".config", "gmlx")):
        os.makedirs(folder)
    with open(os.path.join(home, ".config", "gmlx", "gmlx.yaml"), "w") as f:
        f.write(f"server:\n  host: 127.0.0.1\n  port: {port}\n  menubar: false\n"
                "launch:\n  container:\n    open_browser: false\n  agents:\n"
                f"    {AGENT}:\n      runtime: python\n      command: [e2e-agent]\n"
                f"    {WEB}:\n      runtime: python\n      command: [web.py]\n"
                f"      web_port: {guest_port}\n"
                f"    {SRC}:\n      runtime: python\n      source: {work['agent']}\n"
                "      command: [e2e-agent]\n"
                f"    {BUILD}:\n      runtime: python\n      build: {box}\n"
                f"      source: {work['tools']}\n      command: [build_check.py]\n"
                f"    {API}:\n      runtime: python\n      api: anthropic\n"
                f"      source: {work['tools']}\n      command: [api_check.py]\n")
    os.makedirs(os.path.join(work["agent"], "src", "e2e_agent"))
    files = {(work["agent"], "pyproject.toml"): PYPROJECT,
             (work["agent"], "src/e2e_agent/__init__.py"): AGENT_CODE,
             (work["web"], "web.py"): WEB_CODE, (work["web2"], "web.py"): WEB_CODE,
             (work["tools"], "build_check.py"): BUILD_CODE,
             (work["tools"], "api_check.py"): API_CODE, (box, "Containerfile"): CONTAINERFILE}
    for (folder, name), text in files.items():
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)
    tmp = os.path.join(root, "t")
    os.makedirs(tmp)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GMLX_", "OPENAI_", "ANTHROPIC_"))}
    env.update({"HOME": home, "XDG_DATA_HOME": os.path.join(root, "data"),
                "XDG_CACHE_HOME": os.path.join(root, "cache"),
                "XDG_CONFIG_HOME": os.path.join(home, ".config"), "TMPDIR": tmp,
                "PYTHONPATH": repo + (os.pathsep + env["PYTHONPATH"]
                                      if env.get("PYTHONPATH") else "")})
    return {"home": home, "work": work, "env": env,
            "launch_data": os.path.join(root, "data", "gmlx", "launch")}


def stop_group(proc: subprocess.Popen) -> str:
    """Stop launch and the processes of its group with SIGTERM, so launch
    stops the container it started, and with SIGKILL after 60 seconds or a
    Ctrl-C. When stdin is no terminal, launch starts `container run` in a
    group of its own, which can outlive the SIGKILL and hold the output pipe
    open, so the wait after the SIGKILL is bounded too, and the cleanup
    stops that container. A Ctrl-C during the stop is raised again once
    launch is gone. Return the output read until then, which a Ctrl-C
    after the SIGKILL loses."""
    def kill(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    interrupted = False
    out = None
    kill(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        interrupted = True
    if out is None:
        kill(signal.SIGKILL)
        try:
            out, _ = proc.communicate(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
            interrupted = interrupted or isinstance(e, KeyboardInterrupt)
            # A timeout carries the output read so far, as bytes. After a
            # Ctrl-C that output is lost.
            partial = getattr(e, "output", None)
            out = partial.decode(errors="replace") if isinstance(partial, bytes) else ""
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    if interrupted:
        raise KeyboardInterrupt
    return out or ""


class Background:
    """A launch that runs while the script goes on. A thread collects its
    output, and :meth:`stop` sends SIGTERM to launch alone, as the close of
    its window does, so launch stops its container."""

    def __init__(self, argv: list[str], cwd: str, env: dict, log: str):
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        self._lines: list[str] = []
        self._log = log
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        with open(self._log, "a") as f:
            for line in self.proc.stdout:
                self._lines.append(line)
                f.write(line)
                f.flush()
        # Reap launch as a shell does, since --stop waits until it is gone.
        self.proc.wait()

    @property
    def text(self) -> str:
        return "".join(self._lines)

    def wait_for(self, pattern: str, timeout: float) -> re.Match | None:
        deadline = time.monotonic() + timeout
        while True:
            m = re.search(pattern, self.text)
            if m or time.monotonic() >= deadline:
                return m
            if self.proc.poll() is not None:
                self._thread.join(5)
                return re.search(pattern, self.text)
            time.sleep(0.5)

    def stop(self, timeout: float = 90.0) -> int | None:
        """The exit code of launch after SIGTERM, or None when it outlived
        ``timeout`` and its group got SIGKILL."""
        if self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
                self._thread.join(5)
                return None
        self._thread.join(5)
        return self.proc.returncode


class Run:
    """What the groups share: the options, the scratch HOME, the log, the
    checks, the background launches to stop, and each agent and folder that
    launched, whose project can hold a home and a volume."""

    def __init__(self, a, scratch: dict, log: str, check: Check, guest_port: int):
        self.a, self.scratch, self.log, self.check = a, scratch, log, check
        self.work = scratch["work"]
        self.guest_port = guest_port
        self.sessions: list[Background] = []
        self.detached: list[tuple[str, str]] = []
        self.used: list[tuple[str, str]] = []

    def _argv(self, agent: str, args) -> list[str]:
        return [self.a.python, "-P", "-m", "gmlx", "launch", agent, *args]

    def _use(self, agent: str, cwd: str, args) -> None:
        if (not {"--config-only", "--stop", "--list"} & set(args)
                and (agent, cwd) not in self.used):
            self.used.append((agent, cwd))

    def launch(self, agent: str, *args: str, cwd: str, timeout: float) -> tuple[int, str]:
        """Run ``gmlx launch AGENT ARGS`` in ``cwd`` with no terminal, and
        return its exit code, or -1 after a timeout, and its output. A
        session that ``--detach`` starts is ended with the group."""
        self._use(agent, cwd, args)
        rc, out = self.gmlx("launch", agent, *args, cwd=cwd, timeout=timeout)
        if "--detach" in args[:args.index("--") if "--" in args else len(args)]:
            self.detached.append((agent, cwd))
        return rc, out

    def gmlx(self, *args: str, cwd: str, timeout: float) -> tuple[int, str]:
        """Run ``gmlx ARGS`` in ``cwd`` with no terminal, and return its exit
        code, or -1 after a timeout, and its output."""
        argv = [self.a.python, "-P", "-m", "gmlx", *args]
        t0 = time.monotonic()
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)}\n")
        # A process group of its own, so that a timeout reaches launch and its
        # container run child, and launch stops the container it started.
        proc = subprocess.Popen(argv, cwd=cwd, env=self.scratch["env"], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        shown = f"gmlx {' '.join(args)}"
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            out = stop_group(proc)
            with open(self.log, "a") as f:
                f.write(out + f"\n# timed out after {timeout:.0f}s\n")
            print(f"  {shown} -> timed out after {timeout:.0f}s", flush=True)
            return -1, out
        except BaseException:
            # Launch runs in a session of its own, so a Ctrl-C here does not
            # reach it. It is stopped before the cleanup that removes its home.
            stop_group(proc)
            raise
        out = out or ""
        with open(self.log, "a") as f:
            f.write(out + f"\n# exit {proc.returncode} in {time.monotonic() - t0:.0f}s\n")
        print(f"  {shown} -> exit {proc.returncode} in {time.monotonic() - t0:.0f}s", flush=True)
        return proc.returncode, out

    def background(self, agent: str, *args: str, cwd: str) -> Background:
        argv = self._argv(agent, args)
        self._use(agent, cwd, args)
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)} (in the background)\n")
        session = Background(argv, cwd, self.scratch["env"], self.log)
        self.sessions.append(session)
        return session

    @contextlib.contextmanager
    def pty(self, agent: str, *args: str, cwd: str):
        argv = self._argv(agent, args)
        self._use(agent, cwd, args)
        with open(self.log, "a") as f:
            f.write(f"\n# cd {cwd}; {' '.join(argv)} (pty)\n")
            with PtyProcess(argv, env=self.scratch["env"], log=f, cwd=cwd) as p:
                yield p

    def remove_home(self, agent: str, cwd: str) -> tuple[int | None, str]:
        """Answer yes to --remove-home in a pty. A project whose home is
        removed leaves the list that the cleanup goes through."""
        with self.pty(agent, "--remove-home", cwd=cwd) as p:
            if not p.expect("? [y/N]", timeout=60):
                return p.wait_exit(10), p.transcript
            p.sendline("y")
            rc = p.wait_exit(120)
        if rc == 0 and (agent, cwd) in self.used:
            self.used.remove((agent, cwd))
        return rc, p.transcript

    def kept_ports(self, agent: str) -> set[int]:
        """The Mac ports that the record keeps for the projects of ``agent``."""
        try:
            with open(os.path.join(self.scratch["launch_data"], "web-ports.json")) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return set()
        projects = (doc.get("projects") or {}).get(f"agent-{agent}") or {}
        return {e["port"] for e in projects.values() if isinstance(e, dict) and "port" in e}

    def stop_sessions(self) -> None:
        for session in self.sessions:
            session.stop()
        self.sessions.clear()
        for agent, cwd in self.detached:
            self.gmlx("launch", agent, "--stop", cwd=cwd, timeout=120)
        self.detached.clear()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root)
    ap.add_argument("--model", help="a GGUF path, in place of the tools role of models.py")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that runs the server and gmlx launch")
    ap.add_argument("--out", help="the folder for the logs (default: a fresh temp dir)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch HOME, the agents' homes, their volumes and the images")
    ap.add_argument("--first-timeout", type=float, default=1200.0,
                    help="seconds for a launch that builds an image or installs")
    ap.add_argument("--only", action="append", choices=GROUPS, metavar="GROUP",
                    help=f"run only this group, repeatable: {', '.join(GROUPS)}. join, signals, "
                         "detach, source and doctor run the runtime group first")
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
    groups = list(GROUPS)
    if a.only:
        chosen = set(a.only) | ({"runtime"} if set(a.only) & NEEDS_RUNTIME else set())
        groups = [g for g in GROUPS if g in chosen]

    out = a.out or tempfile.mkdtemp(prefix="gmlx-agents-e2e-")
    os.makedirs(out, exist_ok=True)
    # The real path, since launch refuses a source or build: path that goes
    # through a link, such as /tmp.
    root = os.path.realpath(tempfile.mkdtemp(prefix="gmlx-ae-", dir="/tmp"))
    taken = RESERVED_PORTS | set(WEB_PORTS)
    port = free_port()
    while port in taken:
        port = free_port()
    guest_port = free_port()
    while guest_port in taken | {port}:
        guest_port = free_port()
    scratch = write_scratch(root, port, guest_port, repo)
    log = os.path.join(out, "launch.log")
    images_before = image_names()
    volumes_before = volume_names()
    check = Check()
    run = Run(a, scratch, log, check, guest_port)
    print(f"model {model}\nserver port {port}\nweb_port {guest_port}\nscratch {root}\n"
          f"logs {out}\ngroups {', '.join(groups)}", flush=True)

    # The server keeps its session sockets under the cache folder of its
    # HOME, and launch takes a socket only from the cache folder of its own,
    # so both run in the scratch HOME.
    shared = ("HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR",
              "PYTHONPATH")
    sp = ServerProc([model, "--no-auth"], log_path=os.path.join(out, "server.log"),
                    port=port, python=a.python,
                    env_extra={k: scratch["env"][k] for k in shared})
    try:
        _run_groups(run, sp, groups)
    finally:
        _guarded("stop the background launches", run.stop_sessions, check)
        _guarded("stop the server", sp.stop, check)
        if not a.keep:
            _clean_up(run, root, images_before, volumes_before)
        else:
            print(f"kept {root}, the agents' homes and volumes, and the images", flush=True)

    print(f"\nlogs: {out}")
    if check.failed:
        print("FAILED: " + ", ".join(check.failed))
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _run_groups(run: Run, sp, groups: list[str]) -> None:
    """Start the server, then run each group. An error ends only its own
    group, which then counts as a failed check."""
    try:
        sp.start()
        sp.wait_ready(timeout=900)
    except Exception as e:                                   # noqa: BLE001
        run.check("the server starts", False, f"{type(e).__name__}: {e}")
        return
    print(f"server ready at {sp.base_url}", flush=True)
    for name in groups:
        print(f"\n== {name}", flush=True)
        try:
            GROUP_RUNNERS[name](run)
        except Exception as e:                               # noqa: BLE001
            run.check(f"{name}: the group completes", False, f"{type(e).__name__}: {e}")
        finally:
            run.stop_sessions()


def group_runtime(run: Run) -> None:
    proj = run.work["agent"]
    rc, text = run.launch(AGENT, cwd=proj, timeout=run.a.first_timeout)
    installed = bool(re.search(r"Installed \d+ packages", text))
    run.check("first launch installs the dependencies", installed and rc == 0, f"exit {rc}")
    run.check("the agent completes one tool call through the session socket",
              "E2E_TOOL_CALL add 2 3" in text)
    run.check("the volume and the runtime image are named",
              "at /opt/agent (" in text and f"image {RUNTIME_REPO}:" in text)

    rc, text = run.launch(AGENT, cwd=proj, timeout=600)
    run.check("second launch starts with no download",
              rc == 0 and not re.search(r"Downloading|Installed \d+ packages|Creating virtual",
                                        text) and "E2E_TOOL_CALL add 2 3" in text,
              f"exit {rc}")

    rc, text = run.launch(AGENT, "--network", "none", cwd=proj, timeout=600)
    run.check("a launch under network none starts from the synced volume",
              rc == 0 and "E2E_TOOL_CALL add 2 3" in text
              and "with network none, the client reaches only the gmlx server" in text,
              f"exit {rc}")

    rc, text = run.launch(AGENT, "--", "--exit", "7", cwd=proj, timeout=600)
    run.check("the agent's exit code comes back", rc == 7, f"exit {rc}")


def group_dry_run(run: Run) -> None:
    rc, text = run.launch(WEB, "--config-only", cwd=run.work["web"], timeout=300)
    run.check("the dry run of a web agent maps its web_port, records no port and starts "
              "nothing",
              rc == 0 and f"gmlx-web.sock={run.guest_port}" in text
              and not run.kept_ports(WEB) and not running(WEB), f"exit {rc}")


def _web_up(run: Run, cwd: str, timeout: float) -> tuple[Background, int | None]:
    session = run.background(WEB, cwd=cwd)
    m = session.wait_for(ANSWERS_AT, timeout)
    return session, int(m[1]) if m else None


def group_web(run: Run) -> None:
    guest = run.guest_port
    first, port = _web_up(run, run.work["web"], run.a.first_timeout)
    run.check("a web agent answers at [::1] on a Mac port from 3100 to 3199",
              port in WEB_PORTS, f"port {port}")
    if port is None:
        return
    page = get_json(f"http://[::1]:{port}/") or {}
    run.check("the app listens on its web_port in the guest and gets the [::1] Host",
              page.get("marker") == "E2E_WEB" and page.get("env_host") == "127.0.0.1"
              and page.get("env_port") == str(guest) and page.get("host") == f"[::1]:{port}",
              json.dumps(page))
    status, body = get_with_host(port, f"localhost:{port}")
    run.check("another host name gets the 421 page",
              status == 421 and f"This app answers only at http://[::1]:{port}/" in body,
              f"status {status}")
    closed = {f"127.0.0.1:{port}": refused("127.0.0.1", port),
              f"127.0.0.1:{guest}": refused("127.0.0.1", guest),
              f"[::1]:{guest}": refused("::1", guest)}
    run.check("the Mac answers at no other address of the app", all(closed.values()),
              ", ".join(f"{k} {'closed' if v else 'OPEN'}" for k, v in closed.items()))

    rc, text = run.launch(WEB, cwd=run.work["web"], timeout=300)
    run.check("a second launch from the project names the running app",
              rc == 0 and f"{WEB} is already running at http://[::1]:{port}/" in text
              and len(running(WEB)) == 1, f"exit {rc}")

    second, port2 = _web_up(run, run.work["web2"], 600)
    page2 = (get_json(f"http://[::1]:{port2}/") or {}) if port2 else {}
    run.check("a session of a second project runs at once on a port of its own",
              port2 in WEB_PORTS and port2 != port and page2.get("env_port") == str(guest)
              and (get_json(f"http://[::1]:{port}/") or {}).get("marker") == "E2E_WEB"
              and len(running(WEB)) == 2, f"ports {port} and {port2}")

    exits = [first.stop(), second.stop()]
    gone = wait_until(lambda: not running(WEB), 60)
    run.check("SIGTERM to launch ends each web session and its container",
              None not in exits and gone and refused("::1", port), f"exits {exits}")

    again, port3 = _web_up(run, run.work["web"], 600)
    run.check("the project keeps its port from one launch to the next", port3 == port,
              f"port {port3}, before {port}")
    again.stop()
    wait_until(lambda: not running(WEB), 60)

    with run.pty(WEB, "--shell", cwd=run.work["web"]) as p:
        hint = p.expect("once you start it from the shell with: uv run web.py", 300)
        prompt = p.expect("# ", 120)
        rc, text = run.launch(WEB, cwd=run.work["web"], timeout=300)
        run.check("while the shell holds the session, a second launch names the command",
                  rc == 0 and f"{WEB} answers at http://[::1]:{port}/ once you start it in that "
                  "shell with: uv run web.py" in text, f"exit {rc}")
        p.sendline("uv run web.py")
        listening = p.expect(f"E2E_WEB_LISTENING 127.0.0.1 {guest}", 300)
        shell_page = get_json(f"http://[::1]:{port}/") or {}
        p.send("\x03")
        p.expect("# ", 30)
        # A bare exit would end the shell with the status of the stopped app.
        p.sendline("exit 0")
        rc = p.wait_exit(90)
    run.check("--shell names uv run, and the app that it starts answers at the project's port",
              hint and prompt and listening and shell_page.get("marker") == "E2E_WEB"
              and rc == 0, f"hint {hint}, listening {listening}, "
                           f"page {shell_page.get('marker')}, exit {rc}")
    wait_until(lambda: not running(WEB), 60)

    before = run.kept_ports(WEB)
    rc, text = run.remove_home(WEB, run.work["web"])
    after = run.kept_ports(WEB)
    run.check("--remove-home of a web agent names its site data and releases its port",
              rc == 0 and f"the web app of this project used http://[::1]:{port}" in text
              and "deleted the volume" in text and port in before and port not in after
              and port2 in after, f"exit {rc}, ports before {sorted(before)}, "
                                  f"after {sorted(after)}")


def group_join(run: Run) -> None:
    proj = run.work["agent"]
    with run.pty(AGENT, "--shell", cwd=proj) as shell:
        up = shell.expect("# ", 600)
        rc, text = run.launch(AGENT, cwd=proj, timeout=600)
        count = len(running(AGENT))
        run.check("a second launch joins the running session as another copy",
                  up and rc == 0 and f"joining the running {AGENT} session" in text
                  and "E2E_TOOL_CALL add 2 3" in text and count == 1,
                  f"exit {rc}, {count} containers")
        # The quotes keep the echo of the command line from matching. A uv
        # that made an environment of its own would print another prefix.
        shell.sendline("uv run python -c \"import sys, langchain_openai; "
                       "print('E2E_SHELL' + '_ENV', sys.prefix)\"")
        run.check("uv run in the shell uses the agent's environment",
                  shell.expect("E2E_SHELL_ENV /opt/agent/venv", 120))
        with run.pty(AGENT, "--shell", cwd=proj) as second:
            joined = second.expect(f"opening a shell in the running {AGENT} session", 120)
            prompt = second.expect("# ", 60)
            second.sendline("exit")
            rc2 = second.wait_exit(60)
        count = len(running(AGENT))
        run.check("a second --shell joins the running session",
                  joined and prompt and rc2 == 0 and count == 1,
                  f"exit {rc2}, {count} containers")
        shell.sendline("exit")
        rc3 = shell.wait_exit(90)
    gone = wait_until(lambda: not running(AGENT), 60)
    run.check("the session ends when its shell exits", rc3 == 0 and gone, f"exit {rc3}")


def group_signals(run: Run) -> None:
    proj = run.work["agent"]
    with run.pty(AGENT, "--", "--wait", cwd=proj) as p:
        waiting = p.expect("E2E_WAITING", 600)
        p.send("\x03")
        rc = p.wait_exit(90)
        count = p.transcript.count("E2E_SIGINT")
    gone = wait_until(lambda: not running(AGENT), 60)
    run.check("a Ctrl-C reaches the agent once, and the launch ends with 130",
              waiting and rc == 130 and count == 1 and gone,
              f"exit {rc}, {count} SIGINT lines, container gone {gone}")

    session = run.background(AGENT, "--", "--wait", cwd=proj)
    waiting = bool(session.wait_for("E2E_WAITING", 600))
    rc = session.stop()
    count = session.text.count("E2E_SIGTERM")
    gone = wait_until(lambda: not running(AGENT), 60)
    run.check("SIGTERM to launch reaches the agent once and stops the container",
              waiting and rc is not None and count == 1 and gone,
              f"exit {rc}, {count} SIGTERM lines, container gone {gone}")


def _output_file(run: Run, text: str) -> str | None:
    """The output file that a --detach launch names, with ~ as the scratch
    HOME."""
    m = re.search(OUTPUT_AT, text)
    if not m:
        return None
    return os.path.join(run.scratch["home"], m[1][2:]) if m[1].startswith("~/") else m[1]


def _read(path: str | None) -> str:
    try:
        with open(path or "", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def group_detach(run: Run) -> None:
    web, proj = run.work["web"], run.work["agent"]
    rc, text = run.gmlx("launch", "--list", cwd=web, timeout=60)
    run.check("--list with no session says that none runs",
              rc == 0 and "[launch] no launch session runs." in text, f"exit {rc}")

    rc, text = run.launch(WEB, "--detach", cwd=web, timeout=run.a.first_timeout)
    m = re.search(rf"\[launch\] {WEB} runs in the background for {re.escape(web)} at "
                  r"http://\[::1\]:(\d+)/\.", text)
    port = int(m[1]) if m else None
    page = (get_json(f"http://[::1]:{port}/") or {}) if port else {}
    run.check("--detach of a web agent returns once the app answers, and the app goes on",
              rc == 0 and port in WEB_PORTS and page.get("marker") == "E2E_WEB"
              and len(running(WEB)) == 1, f"exit {rc}, port {port}")
    output = _output_file(run, text)
    run.check("the output file takes the output of the session",
              wait_until(lambda: "E2E_WEB_LISTENING" in _read(output), 30), f"file {output}")
    if port is None:
        return

    rc, text = run.gmlx("launch", "--list", cwd=web, timeout=60)
    row = re.search(rf"^{WEB}\s+{re.escape(web)}\s+running\s+detached\s+"
                    rf"http://\[::1\]:{port}/\s", text, re.M)
    run.check("--list shows the session as detached, with its folder and address",
              rc == 0 and bool(row), f"exit {rc}")
    rc, text = run.gmlx("status", cwd=web, timeout=60)
    run.check("gmlx status names the session",
              f"launch session {WEB} for {web}: running, detached, http://[::1]:{port}/" in text,
              f"exit {rc}")

    rc, text = run.launch(WEB, "--detach", cwd=web, timeout=300)
    run.check("a second --detach names the running app",
              rc == 0 and f"{WEB} is already running at http://[::1]:{port}/" in text
              and len(running(WEB)) == 1, f"exit {rc}")

    rc, text = run.launch(WEB, "--stop", cwd=web, timeout=120)
    gone = wait_until(lambda: not running(WEB), 30)
    run.check("--stop ends the session and its container, and frees the port",
              rc == 0 and f"[launch] stopped the {WEB} session for {web}." in text and gone
              and refused("::1", port), f"exit {rc}, container gone {gone}")
    rc, text = run.launch(WEB, "--stop", cwd=web, timeout=60)
    run.check("a second --stop says that no session runs",
              rc == 0 and f"[launch] no {WEB} session runs for {web}." in text, f"exit {rc}")

    rc, text = run.launch(AGENT, "--detach", "--", "--wait", cwd=proj, timeout=600)
    ran = len(running(AGENT)) == 1                  # as --detach returns
    output = _output_file(run, text)
    waiting = wait_until(lambda: "E2E_WAITING" in _read(output), 300)
    run.check("--detach of an agent returns once its container runs, and the agent goes on",
              rc == 0 and f"[launch] {AGENT} runs in the background for {proj}." in text
              and ran and waiting and len(running(AGENT)) == 1,
              f"exit {rc}, running at return {ran}, waiting {waiting}")
    rc, text = run.launch(AGENT, "--detach", cwd=proj, timeout=120)
    run.check("a second --detach of an agent with no browser interface is refused",
              rc == 1 and f"{AGENT} already runs for {proj}, and --detach starts only a new "
              "session." in text and len(running(AGENT)) == 1, f"exit {rc}")
    rc, text = run.launch(AGENT, "--stop", cwd=proj, timeout=120)
    gone = wait_until(lambda: not running(AGENT), 30)
    count = _read(output).count("E2E_SIGTERM")
    run.check("--stop reaches the detached agent once and stops its container",
              rc == 0 and count == 1 and gone,
              f"exit {rc}, {count} SIGTERM lines, container gone {gone}")

    session = run.background(AGENT, "--", "--wait", cwd=proj)
    waiting = bool(session.wait_for("E2E_WAITING", 600))
    rc, text = run.launch(AGENT, "--stop", cwd=proj, timeout=120)
    ended = wait_until(lambda: session.proc.poll() is not None, 30)
    gone = wait_until(lambda: not running(AGENT), 30)
    count = session.text.count("E2E_SIGTERM")
    run.check("--stop ends a session that a launch without --detach runs",
              waiting and rc == 0 and ended and count == 1 and gone,
              f"exit {rc}, launch ended {ended}, {count} SIGTERM lines, container gone {gone}")

    rc, text = run.gmlx("launch", "pi", "--detach", cwd=web, timeout=60)
    run.check("--detach refuses a client that needs a terminal",
              rc == 1 and "and pi needs a terminal. Launch it without --detach." in text,
              f"exit {rc}")
    rc, text = run.gmlx("launch", WEB, "--detach", "--shell", cwd=web, timeout=60)
    run.check("--detach refuses --shell",
              rc == 2 and "--shell and --detach cannot go together" in text, f"exit {rc}")
    rc, text = run.gmlx("launch", "--list", cwd=web, timeout=60)
    run.check("--list is empty again once the sessions end",
              rc == 0 and "[launch] no launch session runs." in text, f"exit {rc}")


def group_source(run: Run) -> None:
    where, src = run.work["elsewhere"], run.work["agent"]
    rc, text = run.launch(SRC, cwd=where, timeout=run.a.first_timeout)
    shared = re.search(rf"sharing {re.escape(src)} \(read-only, source folder\)", text)
    run.check("a source agent runs from another folder with its source read-only",
              rc == 0 and bool(shared) and "E2E_TOOL_CALL add 2 3" in text, f"exit {rc}")

    probe = os.path.join(src, "e2e-write")
    # The arithmetic keeps the echo of the command from matching.
    rc, text = run.launch(SRC, "--shell", "--", "-c",
                          'touch "$UV_PROJECT/e2e-write" 2>/dev/null && echo E2E_WROTE_$((1+1)) '
                          '|| echo E2E_READ_ONLY_$((1+1))', cwd=where, timeout=300)
    run.check("the agent cannot write to its source",
              rc == 0 and "E2E_READ_ONLY_2" in text and not os.path.exists(probe), f"exit {rc}")

    pyproject = os.path.join(src, "pyproject.toml")
    with open(pyproject) as f:
        original = f.read()
    try:
        with open(pyproject, "w") as f:
            f.write(original.replace('version = "0.1.0"', 'version = "0.1.1"'))
        rc, text = run.launch(SRC, cwd=where, timeout=300)
    finally:
        with open(pyproject, "w") as f:
            f.write(original)
    run.check("a stale uv.lock stops a read-only source with uv's message",
              rc not in (0, -1) and "needs to be updated" in text, f"exit {rc}")


def group_build(run: Run) -> None:
    rc, text = run.launch(BUILD, cwd=run.work["elsewhere"], timeout=run.a.first_timeout)
    run.check("an agent's own build from the runtime base runs a source script with uv",
              rc == 0 and "E2E_BUILD E2E_BUILD_MARKER" in text
              and f"image gmlx.invalid/launch-agent-{BUILD}-build:" in text
              and "has no lockfile, so uv installs the dependencies" in text, f"exit {rc}")


def group_api(run: Run) -> None:
    rc, text = run.launch(API, cwd=run.work["elsewhere"], timeout=600)
    run.check("an api: anthropic agent gets a reply from the Messages route through its "
              "session socket",
              rc == 0 and bool(re.search(r"E2E_API message assistant key gmlx-container-session "
                                         r"openai False window \d+", text)), f"exit {rc}")


def group_doctor(run: Run) -> None:
    argv = [run.a.python, "-P", "-m", "gmlx", "doctor"]
    with open(run.log, "a") as f:
        f.write(f"\n# {' '.join(argv)}\n")
    done = subprocess.run(argv, cwd=run.work["agent"], env=run.scratch["env"],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    text = done.stdout + done.stderr
    with open(run.log, "a") as f:
        f.write(text + f"\n# exit {done.returncode}\n")
    run.check("doctor names the agent's home by the agent name and the project folder",
              f"{AGENT}: {run.work['agent']}, " in text, f"exit {done.returncode}")


GROUP_RUNNERS = {"runtime": group_runtime, "dry-run": group_dry_run, "web": group_web,
                 "join": group_join, "signals": group_signals, "detach": group_detach,
                 "source": group_source, "build": group_build, "api": group_api,
                 "doctor": group_doctor}


def _guarded(what: str, step, check: Check) -> None:
    """Run one cleanup step. A Ctrl-C or an error skips only that step, so
    the steps after it still run, and the skipped step is a failed check."""
    try:
        step()
    except KeyboardInterrupt:
        check(f"cleanup: {what}", False, "skipped after Ctrl-C")
    except Exception as e:                                   # noqa: BLE001
        check(f"cleanup: {what}", False, f"{type(e).__name__}: {e}")


def _clean_up(run: Run, root: str, images_before: set[str], volumes_before: set[str]) -> None:
    """Remove what the run made: a container a timed-out launch left, the
    homes and the volumes through --remove-home, a volume of this run that
    remains, the images and the scratch folder. It runs after a failure and
    after Ctrl-C too, and a Ctrl-C during one step skips only that step."""
    check = run.check

    def ours() -> set[str]:
        """This run's dependency volumes. Each run's project folders are new,
        so their volume names are too."""
        return {v for v in volume_names() - volumes_before if v.startswith("gmlx-agent-e2e-")}

    def containers() -> None:
        for name in _labelled(_rows("ls", "--all"), AGENTS):
            container("stop", "--time", "10", name)
            done = container("delete", name)
            print(f"removed the leftover container {name}" if done.returncode == 0
                  else f"container delete {name} failed: {done.stderr.strip()}", flush=True)

    def homes() -> None:
        for agent, cwd in list(run.used):
            made = {v for v in ours() if v.startswith(f"gmlx-agent-{agent}-uv")}
            rc, text = run.remove_home(agent, cwd)
            if (agent, cwd) == (AGENT, run.work["agent"]):
                left = {v for v in ours() if v.startswith(f"gmlx-agent-{agent}-uv")}
                check("--remove-home removes the home and deletes the dependency volume",
                      rc == 0 and "deleted the volume" in text and bool(made) and not left,
                      f"exit {rc}, volumes before {sorted(made)}")
            else:
                print(f"--remove-home of {agent} in {cwd}: exit {rc}", flush=True)

    def volumes() -> None:
        for name in sorted(ours()):
            done = container("volume", "delete", name)
            print(f"deleted the leftover volume {name}" if done.returncode == 0
                  else f"volume delete {name} failed: {done.stderr.strip()}", flush=True)

    def images() -> None:
        created = sorted(n for n in image_names() - images_before
                         if n.startswith((RUNTIME_REPO, "gmlx.invalid/launch-agent-e2e-")))
        if created:
            done = container("image", "delete", *created)
            print(f"deleted images {created}" if done.returncode == 0
                  else f"image delete failed: {done.stderr.strip()}", flush=True)

    _guarded("remove the leftover containers", containers, check)
    _guarded("--remove-home", homes, check)
    _guarded("delete the leftover volumes", volumes, check)
    _guarded("delete the images", images, check)
    _guarded("remove the scratch folder", lambda: shutil.rmtree(root, ignore_errors=True),
             check)


if __name__ == "__main__":
    sys.exit(main())
