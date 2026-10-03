#!/usr/bin/env python3
"""The flags of Apple's ``container`` command, as its help prints them.

``container_cli.json`` beside this file records, for each command that gmlx
runs, the flags that the real CLI accepts and whether each takes a value.
The fake ``container`` checks every call against it, so a test fails when
gmlx passes a flag that the real CLI would refuse. Run this file with
``--write`` on a Mac with the real CLI to record a new version. The test
``test_the_recorded_flags_match_the_installed_container`` reports when the
record and the installed CLI differ.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "container_cli.json")

# The commands that gmlx and the tests run. ``ls`` is the alias of ``list``.
COMMANDS = (
    "build", "builder start", "builder status", "builder stop", "delete", "exec",
    "image delete", "image inspect", "image list", "image pull", "image tag", "inspect",
    "kill", "list", "logs", "ls", "run", "stop", "system kernel set",
    "system property list", "system start", "system status", "system stop",
    "volume create", "volume delete", "volume inspect", "volume list")

# The commands that pass every word after their first positional argument
# to the process in the container.
PASSTHROUGH = ("run", "exec")

_OPTION = re.compile(r"^  (-[^\s,<]+(?:(?:, |/)-[^\s,<]+)*)( <[^>]+>)?(?:\s|$)")


def parse_help(text: str) -> dict[str, bool]:
    """The flags in the help ``text`` of one command, each with True when
    it takes a value."""
    flags: dict[str, bool] = {}
    for line in text.splitlines():
        m = _OPTION.match(line)
        if m is None:
            continue
        for name in re.split(r", |/", m.group(1)):
            flags[name] = m.group(2) is not None
    return flags


def record(program: str = "container") -> dict:
    """The version and flags of the ``container`` at ``program``."""
    version = subprocess.run([program, "--version"], capture_output=True, text=True,
                             check=True).stdout.split()
    commands = {}
    for command in COMMANDS:
        out = subprocess.run([program, *command.split(), "--help"], capture_output=True,
                             text=True, check=True).stdout
        commands[command] = parse_help(out)
    return {"version": version[version.index("version") + 1], "commands": commands}


def load() -> dict:
    with open(SPEC_PATH) as f:
        return json.load(f)


def problem(args: list[str], spec: dict) -> str | None:
    """Why the real CLI would refuse ``args``, or None. Only the command
    and its flags are checked, not their values."""
    if args in (["--version"], ["--help"]):
        return None
    commands = spec["commands"]
    command = next((c for c in sorted(commands, key=len, reverse=True)
                    if args[:len(c.split())] == c.split()), None)
    if command is None:
        return f"container {' '.join(args[:3])}: a command that gmlx has no record of"
    flags = commands[command]
    rest = args[len(command.split()):]
    i = 0
    while i < len(rest):
        word = rest[i]
        if word == "--":
            break
        if not word.startswith("-") or word == "-":
            if command in PASSTHROUGH:
                break
            i += 1
            continue
        name, eq, _ = word.partition("=")
        if name not in flags:
            return f"container {command}: unknown option {name!r}"
        if flags[name] and not eq:
            if i + 1 >= len(rest):
                return f"container {command}: {name} takes a value"
            i += 1
        i += 1
    return None


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: container_cli_spec.py --write")
    with open(SPEC_PATH, "w") as f:
        json.dump(record(), f, indent=1, sort_keys=True)
        f.write("\n")
