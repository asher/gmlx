#!/usr/bin/env python3
"""A stand-in for Apple's ``container`` command, for the container-mode tests.

The ``fake_container`` fixture installs it on ``PATH`` as ``container``. It
keeps its state in the JSON file named by ``FAKE_CONTAINER_STATE``: the
image store, a registry to pull from, containers, volumes and the result of
each ``gmlx-entry --check``. Every call appends its argv to ``log``. The
output mimics the JSON shapes that apple/container 1.4.1 prints, which
1.5.0 keeps.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import sys
import time

import container_cli_spec


def _normalize(ref: str) -> str:
    """Docker Hub short names, as the real store names them."""
    first = ref.split("/", 1)[0]
    if "/" in ref and ("." in first or ":" in first or first == "localhost"):
        return ref
    return "docker.io/" + (ref if "/" in ref else "library/" + ref)


def _image_json(name: str, img: dict) -> dict:
    variants = []
    for plat in img.get("arch", ["linux/arm64"]):
        os_, arch = plat.split("/")
        inner = {k: v for k, v in (("Entrypoint", img.get("entrypoint")),
                                   ("Cmd", img.get("cmd")),
                                   ("WorkingDir", img.get("workdir"))) if v is not None}
        variants.append({"platform": {"os": os_, "architecture": arch},
                         "size": img.get("size", 0),
                         "config": {"created": img.get("created", "2026-09-01T00:00:00Z"),
                                    "config": inner}})
    variants.append({"platform": {"os": "unknown", "architecture": "unknown"}, "config": {}})
    return {"id": img["digest"][7:], "configuration": {
        "name": name, "descriptor": {"digest": img["digest"], "size": 1}},
        "variants": variants}


def _flag(args: list[str], *names: str) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in names:
            out.append(args[i + 1])
            i += 2
        else:
            i += 1
    return out


def _open_files() -> list[str]:
    """The paths of this process's open file descriptors, so a test can
    check that no lock descriptor reached the child."""
    paths = []
    for fd in range(3, 256):
        try:
            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
        except OSError:
            continue
        paths.append(raw.split(b"\0", 1)[0].decode())
    return paths


def main(state: dict, args: list[str]) -> int:
    state.setdefault("log", []).append(args)
    images = state["images"] = {_normalize(k): v for k, v in state.get("images", {}).items()}
    if args == ["--version"]:
        print(f"container CLI version {state.get('version', '1.5.0')} (build: release)")
        return 0
    if args[:2] == ["system", "status"]:
        # ``app_root`` is the folder of a service that ``container system
        # start --app-root`` started.
        as_json = _flag(args, "--format") == ["json"]
        if state.get("running", True):
            root = state.get("app_root") or os.path.join(
                os.environ["HOME"], "Library", "Application Support", "com.apple.container")
            print(json.dumps({"status": "running", "paths": {
                "appRoot": root.rstrip("/") + "/", "installRoot": "/usr/local/"}})
                  if as_json else "FIELD   VALUE\nstatus  running")
            return 0
        if as_json:
            print(json.dumps({"status": "not running"}))
        else:
            print("apiserver is not running", file=sys.stderr)
        return 1
    if args[:2] == ["system", "start"]:
        # The service starts before the kernel step. A start with neither
        # kernel flag would ask a question on the terminal, which launch
        # never lets it do. ``start_down`` fails the start before the
        # service answers, and ``kernel_fail`` and ``kernel_interrupt`` act
        # as for ``system kernel set``.
        if "--enable-kernel-install" not in args and "--disable-kernel-install" not in args:
            print("fake container: a start without a kernel flag asks a question",
                  file=sys.stderr)
            return 64
        if state.get("start_down"):
            print("Error: failed to get a response from apiserver", file=sys.stderr)
            return 1
        state["running"] = True
        state.pop("app_root", None)
        _read_settings(state)
        if "--enable-kernel-install" in args:
            return _kernel_download(state)
        return 0
    if args[:3] == ["system", "kernel", "set"]:
        if not state.get("running", True):
            print("Error: failed to get a response from apiserver", file=sys.stderr)
            return 1
        return _kernel_download(state)
    if args[:2] == ["system", "stop"]:
        state["running"] = False
        return 0
    if args[:3] == ["system", "property", "list"]:
        # ``rosetta`` is the builder's setting that the service read when it
        # started.
        print(json.dumps({"build": {"cpus": 2, "memory": "2048mb",
                                    "rosetta": state.get("rosetta", True)},
                          "container": {"cpus": 4, "memory": "1gb"}}))
        return 0
    if args[:2] == ["image", "inspect"]:
        if state.get("inspect_error"):
            print(f"Error: {state['inspect_error']}", file=sys.stderr)
            return 1
        img = images.get(_normalize(args[2]))
        if img is None:
            print(f"Error: image not found: {args[2]}", file=sys.stderr)
            return 1
        print(json.dumps([_image_json(_normalize(args[2]), img)]))
        return 0
    if args[:2] == ["image", "list"]:
        print(json.dumps([_image_json(n, i) for n, i in sorted(images.items())]))
        return 0
    if args[:2] == ["image", "tag"]:
        if _normalize(args[2]) not in images:
            return 1
        images[_normalize(args[3])] = dict(images[_normalize(args[2])])
        return 0
    if args[:2] == ["image", "delete"]:
        refused = [ref for ref in args[2:] if ref in state.get("refuse_delete", [])]
        for ref in args[2:]:
            if ref not in refused:
                images.pop(_normalize(ref), None)
        state.setdefault("deleted", []).extend(args[2:])
        if refused:
            print(f"Error: failed to delete one or more images: {refused}", file=sys.stderr)
            return 1
        return 0
    if args[:2] == ["image", "pull"]:
        img = state.get("registry", {}).get(args[2])
        if img is None:
            print(f"Error: {args[2]} not found in the registry", file=sys.stderr)
            return 1
        images[_normalize(args[2])] = dict(img)
        return 0
    if args[:2] == ["builder", "status"]:
        # With no builder at all, 1.4.1 prints an empty list.
        if state.get("builder") is None:
            print("[]")
            return 0
        conf = state.get("builder_config", {"cpus": 2, "memory": 2 << 30, "ssh": False})
        env = ["PATH=/usr/bin:/bin", *state.get("builder_env", [])]
        if state.get("builder_stopping"):
            # A builder that is stopping reports it this many more times.
            state["builder_stopping"] -= 1
            builder_state = "stopping"
        else:
            builder_state = "running" if state["builder"] else "stopped"
        print(json.dumps([{"id": "buildkit", "configuration": {
            "id": "buildkit", "ssh": conf["ssh"], "initProcess": {"environment": env},
            "resources": {"cpus": conf["cpus"], "memoryInBytes": conf["memory"]}},
            "status": {"state": builder_state,
                       "startedDate": state.get("builder_started", "2026-01-01T00:00:00Z")}}]))
        return 0
    if args[:2] == ["builder", "stop"]:
        if state.get("fail_builder_stop"):
            print("Error: internalError: the builder did not stop", file=sys.stderr)
            return 1
        if state.get("builder") is not None:
            state["builder"] = False
        return 0
    if args[0] == "build":
        # As 1.4.1 does, the builder takes BUILDKIT_COLORS and NO_COLOR from
        # the build command, and a running builder whose copy differs is
        # created again.
        wanted = sorted([f"BUILDKIT_COLORS={os.environ['BUILDKIT_COLORS']}"]
                        if "BUILDKIT_COLORS" in os.environ else []) + (
            ["NO_COLOR=true"] if "NO_COLOR" in os.environ else [])
        wanted.sort()
        if state.get("builder") and state.get("builder_env", []) != wanted:
            state["builder_recreated"] = state.get("builder_recreated", 0) + 1
            state["builder"] = False
        if not state.get("builder"):
            # Each start gets its own start date, as whole seconds would
            # not tell two quick starts apart.
            # ``real_clock`` gives the time of the start, as the service does.
            state["builder_starts"] = state.get("builder_starts", 0) + 1
            state["builder_started"] = f"2026-09-27T12:{state['builder_starts'] // 60:02d}:" \
                                       f"{state['builder_starts'] % 60:02d}Z"
            if state.get("real_clock"):
                state["builder_started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["builder_env"] = wanted
        state["builder"] = True
        state.setdefault("builder_args", []).append(
            [a for i, a in enumerate(args) if a in ("--cpus", "--memory", "--ssh")
             or (i and args[i - 1] in ("--cpus", "--memory", "--ssh"))])
        if state.get("build_waits_for_resize"):
            # Report the size of standard error after the first SIGWINCH, or
            # after 5 s when none arrives.
            wanted = {signal.SIGWINCH, signal.SIGALRM}
            # macOS drops a signal whose action is to ignore it, as SIGWINCH's
            # default is, even during sigwait, so each gets a handler first.
            for sig in wanted:
                signal.signal(sig, lambda *_: None)
            signal.pthread_sigmask(signal.SIG_BLOCK, wanted)
            print("waiting for a resize", file=sys.stderr, flush=True)
            signal.alarm(5)
            got = signal.sigwait(wanted)
            size = os.get_terminal_size(2)
            print(f"got {signal.Signals(got).name} at {size.columns}x{size.lines}",
                  file=sys.stderr, flush=True)
        if state.get("fail_build"):
            # A string is the output of the failed step, and the fake says
            # whether its standard error was a terminal.
            if isinstance(state["fail_build"], str):
                print(f"{state['fail_build']}\nstderr tty: {sys.stderr.isatty()}",
                      file=sys.stderr)
            return 1
        state["next"] = state.get("next", 0) + 1
        digest = "sha256:" + hashlib.sha256(str(state["next"]).encode()).hexdigest()
        tags = _flag(args, "--tag")
        build_args = dict(a.split("=", 1) for a in _flag(args, "--build-arg"))
        state.setdefault("builds", []).append({
            "tags": tags, "build_args": build_args, "file": _flag(args, "--file")[0],
            "context": args[-1], "no_cache": "--no-cache" in args, "pull": "--pull" in args})
        for tag in tags:
            images[_normalize(tag)] = {"digest": digest, "created": state.get("now", "2026-09-27T00:00:00Z")}
        return 0
    if args[0] == "run" and "--check" not in args:
        names = [args[i + 1] for i, a in enumerate(args[:-1])
                 if a == "-e" and "=" not in args[i + 1]]
        state.setdefault("runs", []).append(
            {"argv": args, "env": {n: os.environ.get(n) for n in names},
             "pgid": os.getpgrp(), "open_files": _open_files()})
        return state.get("run_rc", 0)
    if args[0] == "run":
        word = args[args.index("--check") + 1] if "--check" in args else ""
        rc, line = state.get("checks", {}).get(word, [0, f"/usr/bin/{word}"])
        print(line, file=sys.stderr if rc else sys.stdout)
        return rc
    if args[0] == "ls":
        print(json.dumps([{
            "id": c["name"], "status": {"state": c.get("state", "running")},
            "configuration": {
                "id": c["name"], "labels": c.get("labels", {}),
                "image": {"reference": c.get("image", ""),
                          "descriptor": {"digest": c.get("image_digest", "")}},
                "mounts": [{"source": "", "destination": "/v", "options": [],
                            "type": {"volume": {"name": v, "format": "ext4"}}}
                           for v in c.get("volumes", [])],
                "resources": {"cpus": 4, "memoryInBytes": c.get("memory", 4 << 30)}}}
            for c in state.get("containers", [])]))
        return 0
    if args[0] in ("stop", "kill", "delete"):
        # With ``delete_removes`` a delete takes the container off the list.
        if args[0] == "delete" and state.get("delete_removes"):
            state["containers"] = [c for c in state.get("containers", [])
                                   if c["name"] != args[-1]]
        return 0
    if args[0] == "exec" and "--hangup" in args:
        # The guest copy is a process on the Mac here: the pid in the file
        # that ``hangup_kill`` names. ``hangup_notify`` names a FIFO that
        # hears of each hangup.
        if state.get("hangup_kill"):
            with open(state["hangup_kill"]) as f:
                os.kill(int(f.read()), signal.SIGHUP)
        if state.get("hangup_notify"):
            with open(state["hangup_notify"], "w") as f:
                f.write("hangup\n")
        return 0
    if args[:2] == ["volume", "list"]:
        print(json.dumps([{"id": v["name"], "configuration": {
            "name": v["name"], "labels": v.get("labels", {}),
            "options": {"size": v["size"]} if v.get("size") else {},
            "sizeInBytes": v.get("bytes", 549755813888), "source": v.get("source", ""),
            "format": "ext4"}} for v in state.get("volumes", [])]))
        return 0
    if args[:2] == ["volume", "create"]:
        labels = dict(a.split("=", 1) for a in _flag(args, "--label"))
        size = (_flag(args, "-s") or [None])[0]
        volume = {"name": args[-1], "labels": labels, "size": size}
        if size:
            volume["bytes"] = int(size[:-1]) << {"K": 10, "M": 20, "G": 30, "T": 40}[size[-1]]
        state.setdefault("volumes", []).append(volume)
        return 0
    if args[:2] == ["volume", "delete"]:
        name = args[2]
        volumes = state.get("volumes", [])
        if name in state.get("refuse_volume_delete", []) or not any(
                v["name"] == name for v in volumes):
            print(f"Error: volume {name} could not be deleted", file=sys.stderr)
            return 1
        state["volumes"] = [v for v in volumes if v["name"] != name]
        return 0
    print(f"fake container: unhandled {args}", file=sys.stderr)
    return 64


def _read_settings(state: dict) -> None:
    """As the service does when it starts: take the builder's Rosetta
    setting from the user file of the test's HOME."""
    import tomllib

    path = os.path.join(os.environ["HOME"], ".config", "container", "config.toml")
    try:
        with open(path, "rb") as f:
            build = tomllib.load(f).get("build") or {}
    except (OSError, tomllib.TOMLDecodeError):
        return
    if "rosetta" in build:
        state["rosetta"] = build["rosetta"]


def _kernel_download(state: dict) -> int:
    """The download of the recommended kernel. ``kernel_fail`` fails it, as
    a download with no network does, and ``kernel_interrupt`` sends the
    launch that ran the call a SIGINT, as a Ctrl-C on the terminal does,
    and ends with the exit code of SIGINT."""
    print("Installing kernel...", file=sys.stderr)
    if state.get("kernel_interrupt"):
        os.kill(os.getppid(), signal.SIGINT)
        return 130
    if state.get("kernel_fail"):
        print("Error: failed to download the kernel: network is unreachable", file=sys.stderr)
        return 1
    _install_kernel(state.get("app_root"))
    state["kernel_installs"] = state.get("kernel_installs", 0) + 1
    return 0


def _install_kernel(root: str | None = None) -> None:
    """Put a kernel where Apple container keeps it: in the folder of a
    service started with --app-root, or for the test's HOME. A HOME that is
    the account's own is never written."""
    import pwd

    home = os.environ["HOME"]
    if root is None and os.path.realpath(home) == os.path.realpath(
            pwd.getpwuid(os.getuid()).pw_dir):
        return
    base = root or os.path.join(home, "Library", "Application Support", "com.apple.container")
    kernels = os.path.join(base, "kernels")
    os.makedirs(kernels, exist_ok=True)
    with open(os.path.join(kernels, "default.kernel-arm64"), "wb") as f:
        f.write(b"kernel")


def _call_name(args: list[str]) -> str:
    return "hangup" if args[:1] == ["exec"] and "--hangup" in args else (args or [""])[0]


def _event(folder: str, line: str) -> None:
    # A test that has stopped reading leaves no reader, and the call goes on.
    try:
        fd = os.open(os.path.join(folder, "events"), os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def _watch(folder: str, name: str) -> None:
    """With ``FAKE_CONTAINER_WATCH`` set to a folder, each call reports on
    the FIFO ``events`` there when it starts, with 1 when it leads a process
    group of its own, and its exit code when it ends. A SIGHUP ends the call,
    as it ends the real CLI, which has no handler for it, and the call
    reports it first. A call whose name is a file in the folder takes that
    file and then waits for a line on the FIFO ``release``, so that a test
    can send signals while the call runs."""
    def hung_up(signum, frame):
        _event(folder, f"signalled {name}")
        os._exit(128 + signal.SIGHUP)
    signal.signal(signal.SIGHUP, hung_up)
    _event(folder, f"start {name} {int(os.getpgrp() == os.getpid())}")
    try:
        os.unlink(os.path.join(folder, name))
    except FileNotFoundError:
        return
    with open(os.path.join(folder, "release")) as f:
        f.readline()


if __name__ == "__main__":
    path = os.environ["FAKE_CONTAINER_STATE"]
    watch = os.environ.get("FAKE_CONTAINER_WATCH")
    if watch:
        _watch(watch, _call_name(sys.argv[1:]))
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(path) as f:
            current = json.load(f)
        # The real CLI refuses a flag that its help does not list, with
        # exit 64. The fixture fails the test that made such a call.
        refused = container_cli_spec.problem(sys.argv[1:], container_cli_spec.load())
        if refused is not None:
            current.setdefault("refused", []).append(refused)
            print(f"Error: {refused}", file=sys.stderr)
            code = 64
        else:
            code = main(current, sys.argv[1:])
        with open(path, "w") as f:
            json.dump(current, f)
    if watch:
        _event(watch, f"end {_call_name(sys.argv[1:])} {code}")
    sys.exit(code)
