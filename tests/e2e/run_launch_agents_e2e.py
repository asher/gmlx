#!/usr/bin/env python3
"""Run a custom agent from launch.agents in a real Apple container.

The script writes a user config in a scratch HOME that defines one runtime
agent, a small LangChain project that makes one tool call, and launches it
through ``gmlx launch`` against a server it starts itself on a free port. It
checks that the first launch installs the dependencies and the agent
completes the tool call through the session socket, that the second launch
starts with no download, that a launch under ``network: none`` starts from
the synced volume, and that the agent's exit code comes back. At the end it
answers yes to ``--remove-home`` in a pty, which removes the agent's home
and its dependency volume, and deletes the images the run created.

    python tests/e2e/run_launch_agents_e2e.py

It needs Apple container 1.5.0 or newer with its service running, the guest
entry from ``scripts/build_guest_entry.py``, network access for the first
install, and an official model that calls tools under the models root. It
prints SKIP and exits 0 when one of them is missing. The real
``~/.config/gmlx/gmlx.yaml`` is never read or edited, and the server never
uses port 8091 or 8092. Exit status 0 means every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ModelRegistry  # noqa: E402
from pty_session import PtyProcess  # noqa: E402
from server_proc import ServerProc, free_port  # noqa: E402

AGENT = "e2e-agent"
RESERVED_PORTS = {8091, 8092}
RUNTIME_REPO = "gmlx.invalid/launch-runtime-python"

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

AGENT_CODE = '''"""The agent of the launch-agents end-to-end script: one tool call."""
import os
import sys

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main():
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


def image_names() -> set[str]:
    out = container("image", "list", "--format", "json").stdout
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError:
        return set()
    return {(row.get("configuration") or row).get("name", "") for row in rows} - {""}


def volume_names() -> set[str]:
    out = container("volume", "list", "--format", "json").stdout
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError:
        return set()
    return {(r.get("configuration") or r).get("name", "") for r in rows}


def agent_containers(scratch: dict) -> list[str]:
    """The containers, running or not, of the agent's projects in this
    scratch HOME, found by the labels launch gives them."""
    projects_dir = os.path.join(scratch["env"]["XDG_DATA_HOME"], "gmlx", "launch",
                                f"agent-{AGENT}", "projects")
    try:
        projects = set(os.listdir(projects_dir))
    except OSError:
        projects = set()
    out = container("ls", "--all", "--format", "json").stdout
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError:
        return []
    names = []
    for row in rows:
        conf = row.get("configuration") or {}
        labels = conf.get("labels") or {}
        if (labels.get("gmlx.launch.client") == f"agent-{AGENT}"
                and labels.get("gmlx.launch.project") in projects):
            names.append(row.get("id") or conf.get("id", ""))
    return [n for n in names if n]


def write_scratch(root: str, port: int, repo: str) -> dict:
    """The scratch HOME with a user config of one runtime agent, the launch
    state folders, and the agent's project, which is the working folder.
    PYTHONPATH names the checkout, so ``-m gmlx`` runs its code."""
    home = os.path.join(root, "home")
    cfg_dir = os.path.join(home, ".config", "gmlx")
    os.makedirs(cfg_dir)
    with open(os.path.join(cfg_dir, "gmlx.yaml"), "w") as f:
        f.write(f"server:\n  host: 127.0.0.1\n  port: {port}\n  menubar: false\n"
                f"launch:\n  container:\n    open_browser: false\n  agents:\n    {AGENT}:\n"
                "      runtime: python\n      command: [e2e-agent]\n")
    project = os.path.join(root, "work", "agent")
    os.makedirs(os.path.join(project, "src", "e2e_agent"))
    with open(os.path.join(project, "pyproject.toml"), "w") as f:
        f.write(PYPROJECT)
    with open(os.path.join(project, "src", "e2e_agent", "__init__.py"), "w") as f:
        f.write(AGENT_CODE)
    tmp = os.path.join(root, "t")
    os.makedirs(tmp)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GMLX_", "OPENAI_"))}
    env.update({"HOME": home, "XDG_DATA_HOME": os.path.join(root, "data"),
                "XDG_CACHE_HOME": os.path.join(root, "cache"),
                "XDG_CONFIG_HOME": os.path.join(home, ".config"), "TMPDIR": tmp,
                "PYTHONPATH": repo + (os.pathsep + env["PYTHONPATH"]
                                      if env.get("PYTHONPATH") else "")})
    return {"home": home, "project": project, "env": env}


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


def launch(scratch: dict, python: str, *args: str, log: str, timeout: float
           ) -> tuple[int, str]:
    """Run ``gmlx launch <agent> ARGS`` from the project folder and return
    its exit code and output."""
    argv = [python, "-P", "-m", "gmlx", "launch", AGENT, *args]
    t0 = time.monotonic()
    with open(log, "a") as f:
        f.write(f"\n# {' '.join(argv)}\n")
    # A process group of its own, so that a timeout reaches launch and its
    # container run child, and launch stops the container it started.
    proc = subprocess.Popen(argv, cwd=scratch["project"], env=scratch["env"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        out = stop_group(proc)
        with open(log, "a") as f:
            f.write(out + f"\n# timed out after {timeout:.0f}s\n")
        print(f"  gmlx launch {AGENT} {' '.join(args)} -> timed out after {timeout:.0f}s",
              flush=True)
        return -1, out
    except BaseException:
        # Launch runs in a session of its own, so a Ctrl-C here does not
        # reach it. It is stopped before the cleanup that removes its home.
        stop_group(proc)
        raise
    out = out or ""
    with open(log, "a") as f:
        f.write(out + f"\n# exit {proc.returncode} in {time.monotonic() - t0:.0f}s\n")
    print(f"  gmlx launch {AGENT} {' '.join(args)} -> exit {proc.returncode} in "
          f"{time.monotonic() - t0:.0f}s", flush=True)
    return proc.returncode, out


def remove_home(scratch: dict, python: str, log: str) -> tuple[int | None, str]:
    """Answer yes to --remove-home in a pty."""
    argv = [python, "-P", "-m", "gmlx", "launch", AGENT, "--remove-home"]
    os.chdir(scratch["project"])
    with open(log, "a") as f:
        f.write(f"\n# {' '.join(argv)} (pty)\n")
        with PtyProcess(argv, env=scratch["env"], log=f) as p:
            if not p.expect("? [y/N]", timeout=60):
                return p.wait_exit(10), p.transcript
            p.sendline("y")
            return p.wait_exit(120), p.transcript


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", default=ModelRegistry.root)
    ap.add_argument("--model", help="a GGUF path, in place of the tools role of models.py")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that runs the server and gmlx launch")
    ap.add_argument("--out", help="the folder for the logs (default: a fresh temp dir)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch HOME, the agent's home, its volume and the images")
    ap.add_argument("--first-timeout", type=float, default=1200.0,
                    help="seconds for the first launch, which builds and installs")
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

    out = a.out or tempfile.mkdtemp(prefix="gmlx-agents-e2e-")
    os.makedirs(out, exist_ok=True)
    root = tempfile.mkdtemp(prefix="gmlx-ae-", dir="/tmp")
    port = free_port()
    while port in RESERVED_PORTS:
        port = free_port()
    scratch = write_scratch(root, port, repo)
    log = os.path.join(out, "launch.log")
    images_before = image_names()
    volumes_before = volume_names()
    check = Check()
    print(f"model {model}\nserver port {port}\nscratch {root}\nlogs {out}", flush=True)

    # The server keeps its session sockets under the cache folder of its
    # HOME, and launch takes a socket only from the cache folder of its own,
    # so both run in the scratch HOME.
    shared = ("HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR",
              "PYTHONPATH")
    sp = ServerProc([model, "--no-auth"], log_path=os.path.join(out, "server.log"),
                    port=port, python=a.python,
                    env_extra={k: scratch["env"][k] for k in shared})
    try:
        _run_checks(a, scratch, sp, log, check)
    finally:
        _guarded("stop the server", sp.stop, check)
        if not a.keep:
            _clean_up(a, scratch, root, log, images_before, volumes_before, check)
        else:
            print(f"kept {root}, the agent's home and volume, and the images", flush=True)

    print(f"\nlogs: {out}")
    if check.failed:
        print("FAILED: " + ", ".join(check.failed))
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _run_checks(a, scratch: dict, sp, log: str, check: Check) -> None:
    """Start the server and run the four launches."""
    try:
        sp.start()
        sp.wait_ready(timeout=900)
        print(f"server ready at {sp.base_url}", flush=True)

        rc, text = launch(scratch, a.python, log=log, timeout=a.first_timeout)
        installed = bool(re.search(r"Installed \d+ packages", text))
        check("first launch installs the dependencies", installed and rc == 0,
              f"exit {rc}")
        check("the agent completes one tool call through the session socket",
              "E2E_TOOL_CALL add 2 3" in text)
        check("the volume and the runtime image are named",
              "at /opt/agent (" in text and f"image {RUNTIME_REPO}:" in text)

        rc, text = launch(scratch, a.python, log=log, timeout=600)
        check("second launch starts with no download",
              rc == 0 and not re.search(r"Downloading|Installed \d+ packages|Creating virtual",
                                        text) and "E2E_TOOL_CALL add 2 3" in text,
              f"exit {rc}")

        rc, text = launch(scratch, a.python, "--network", "none", log=log, timeout=600)
        check("a launch under network none starts from the synced volume",
              rc == 0 and "E2E_TOOL_CALL add 2 3" in text
              and "with network none, the client reaches only the gmlx server" in text,
              f"exit {rc}")

        rc, text = launch(scratch, a.python, "--", "--exit", "7", log=log, timeout=600)
        check("the agent's exit code comes back", rc == 7, f"exit {rc}")
    except Exception as e:                                   # noqa: BLE001
        check("the run completes", False, f"{type(e).__name__}: {e}")


def _guarded(what: str, step, check: Check) -> None:
    """Run one cleanup step. A Ctrl-C or an error skips only that step, so
    the steps after it still run, and the skipped step is a failed check."""
    try:
        step()
    except KeyboardInterrupt:
        check(f"cleanup: {what}", False, "skipped after Ctrl-C")
    except Exception as e:                                   # noqa: BLE001
        check(f"cleanup: {what}", False, f"{type(e).__name__}: {e}")


def _clean_up(a, scratch: dict, root: str, log: str, images_before: set[str],
              volumes_before: set[str], check: Check) -> None:
    """Remove what the run made: a container a timed-out launch left, the
    home and the volume through --remove-home, a volume of this run that
    remains, the images and the scratch folder. It runs after a failure and
    after Ctrl-C too, and a Ctrl-C during one step skips only that step."""
    def ours() -> set[str]:
        """This run's dependency volumes. Each run's project folder is new,
        so its volume name is too."""
        return {v for v in volume_names() - volumes_before
                if v.startswith(f"gmlx-agent-{AGENT}-uv")}

    def containers() -> None:
        for name in agent_containers(scratch):
            container("stop", "--time", "10", name)
            done = container("delete", name)
            print(f"removed the leftover container {name}" if done.returncode == 0
                  else f"container delete {name} failed: {done.stderr.strip()}", flush=True)

    def home() -> None:
        made = ours()
        rc, text = remove_home(scratch, a.python, log)
        left = ours()
        check("--remove-home removes the home and deletes the dependency volume",
              rc == 0 and "deleted the volume" in text and bool(made) and not left,
              f"exit {rc}, volumes before {sorted(made)}")

    def volumes() -> None:
        for name in sorted(ours()):
            done = container("volume", "delete", name)
            print(f"deleted the leftover volume {name}" if done.returncode == 0
                  else f"volume delete {name} failed: {done.stderr.strip()}", flush=True)

    def images() -> None:
        created = sorted(n for n in image_names() - images_before
                         if n.startswith(("gmlx.invalid/launch-runtime-python",
                                          f"gmlx.invalid/launch-agent-{AGENT}")))
        if created:
            done = container("image", "delete", *created)
            print(f"deleted images {created}" if done.returncode == 0
                  else f"image delete failed: {done.stderr.strip()}", flush=True)

    _guarded("remove the leftover containers", containers, check)
    _guarded("--remove-home", home, check)
    _guarded("delete the leftover volumes", volumes, check)
    _guarded("delete the images", images, check)
    _guarded("remove the scratch folder", lambda: shutil.rmtree(root, ignore_errors=True),
             check)

if __name__ == "__main__":
    sys.exit(main())
