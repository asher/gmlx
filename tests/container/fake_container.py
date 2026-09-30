#!/usr/bin/env python3
"""A stand-in for Apple's ``container`` command, for the container-mode tests.

The ``fake_container`` fixture installs it on ``PATH`` as ``container``. It
keeps its state in the JSON file named by ``FAKE_CONTAINER_STATE``: the
image store, a registry to pull from, containers, volumes and the result of
each ``gmlx-entry --check``. Every call appends its argv to ``log``. The
output mimics the JSON shapes that apple/container 1.4.1 prints.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys


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
        print(f"container CLI version {state.get('version', '1.4.1')} (build: release)")
        return 0
    if args[:2] == ["system", "status"]:
        if state.get("running", True):
            print("FIELD   VALUE\nstatus  running")
            return 0
        print("apiserver is not running", file=sys.stderr)
        return 1
    if args[:2] == ["system", "start"]:
        state["running"] = True
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
            state["builder_starts"] = state.get("builder_starts", 0) + 1
            state["builder_started"] = f"2026-09-27T12:{state['builder_starts'] // 60:02d}:" \
                                       f"{state['builder_starts'] % 60:02d}Z"
        state["builder_env"] = wanted
        state["builder"] = True
        state.setdefault("builder_args", []).append(
            [a for i, a in enumerate(args) if a in ("--cpus", "--memory", "--ssh")
             or (i and args[i - 1] in ("--cpus", "--memory", "--ssh"))])
        if state.get("fail_build"):
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
    print(f"fake container: unhandled {args}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    path = os.environ["FAKE_CONTAINER_STATE"]
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(path) as f:
            current = json.load(f)
        code = main(current, sys.argv[1:])
        with open(path, "w") as f:
            json.dump(current, f)
    sys.exit(code)
