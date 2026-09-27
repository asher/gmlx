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
        img = images.get(_normalize(args[2]))
        if img is None:
            print(f"Error: image {args[2]} not found", file=sys.stderr)
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
        for ref in args[2:]:
            images.pop(_normalize(ref), None)
        state.setdefault("deleted", []).extend(args[2:])
        return 0
    if args[:2] == ["image", "pull"]:
        img = state.get("registry", {}).get(args[2])
        if img is None:
            print(f"Error: {args[2]} not found in the registry", file=sys.stderr)
            return 1
        images[_normalize(args[2])] = dict(img)
        return 0
    if args[0] == "build":
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
            {"argv": args, "env": {n: os.environ.get(n) for n in names}})
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
        state.setdefault("volumes", []).append(
            {"name": args[-1], "labels": labels, "size": size})
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
