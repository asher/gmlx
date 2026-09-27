#!/usr/bin/env python3
"""Build the guest entry of `gmlx launch --container`.

The default build compiles crates/gmlx-entry for aarch64-unknown-linux-musl
in release mode and writes a static binary to gmlx/container/guest/gmlx-entry
with mode 0755. `--native` builds the same crate for this machine instead,
for the entry tests, and prints the binary's path.

Every build first checks the toolchain: `cargo` and `rustc`, run from the
crate folder, must print the version rust-toolchain.toml pins, and the musl
build also needs that target's standard library. Builds use only the
vendored crates, so they need no network.

  python scripts/build_guest_entry.py            # the static guest binary
  python scripts/build_guest_entry.py --native   # a binary for this machine

Stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "gmlx-entry"
TARGET = "aarch64-unknown-linux-musl"
OUT = ROOT / "gmlx" / "container" / "guest" / "gmlx-entry"
INSTALL_HINT = ("Install the pinned toolchain as CONTRIBUTING.md describes:\n"
                "  brew install rustup\n"
                "  export PATH=\"$(brew --prefix rustup)/bin:$PATH\"\n"
                "  rustup toolchain install {version} --profile minimal "
                f"--target {TARGET}")


class ToolchainError(RuntimeError):
    """The toolchain on PATH cannot build the crate."""


def pinned_version(crate: Path = CRATE) -> str:
    """The exact compiler version rust-toolchain.toml pins."""
    text = (crate / "rust-toolchain.toml").read_text()
    m = re.search(r'^channel\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise ToolchainError(f"{crate / 'rust-toolchain.toml'} pins no channel")
    return m.group(1)


def _env() -> dict:
    env = dict(os.environ)
    env["RUSTUP_AUTO_INSTALL"] = "0"    # a missing toolchain is an error, not a download
    return env


def _run_version(tool: str, crate: Path) -> str:
    try:
        done = subprocess.run([tool, "--version"], cwd=crate, env=_env(),
                              capture_output=True, text=True, timeout=60)
    except OSError as e:
        raise ToolchainError(f"{tool} is not on PATH ({e})")
    if done.returncode != 0:
        raise ToolchainError(f"`{tool} --version` failed: "
                             f"{(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def check_toolchain(*, musl: bool, crate: Path = CRATE) -> None:
    """Raise :class:`ToolchainError` unless cargo and rustc match the pinned
    version and, for the musl build, the target's standard library exists."""
    version = pinned_version(crate)
    hint = INSTALL_HINT.format(version=version)
    for tool in ("cargo", "rustc"):
        try:
            line = _run_version(tool, crate)
        except ToolchainError as e:
            raise ToolchainError(f"{e}\n{hint}")
        words = line.split()
        if len(words) < 2 or words[1] != version:
            raise ToolchainError(f"{tool} is {line!r}, but the crate pins {version}.\n{hint}")
    if musl:
        done = subprocess.run(["rustc", "--print", "sysroot"], cwd=crate, env=_env(),
                              capture_output=True, text=True, timeout=60)
        lib = Path(done.stdout.strip()) / "lib" / "rustlib" / TARGET
        if done.returncode != 0 or not lib.is_dir():
            raise ToolchainError(f"rustc {version} has no {TARGET} standard library "
                                 f"({lib} is missing).\n{hint}")


def _rustflags(crate: Path) -> str:
    """Path remaps, so the binary never names a folder of the build machine."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    return " ".join([f"--remap-path-prefix={crate}=/gmlx-entry",
                     f"--remap-path-prefix={cargo_home}=/cargo"])


def _cargo_build(args: list, crate: Path) -> None:
    env = _env()
    env["RUSTFLAGS"] = _rustflags(crate)
    subprocess.run(["cargo", "build", "--release", "--locked", "--offline", *args],
                   cwd=crate, env=env, check=True)


def check_static_aarch64_elf(path: Path) -> None:
    """Raise ValueError unless ``path`` is a 64-bit little-endian aarch64 ELF
    executable with no program interpreter, which means it is static."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{path} is not a 64-bit little-endian ELF file")
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    if e_machine != 183:                        # EM_AARCH64
        raise ValueError(f"{path} is ELF machine {e_machine}, not aarch64 (183)")
    if e_type not in (2, 3):                    # ET_EXEC, or ET_DYN for static-pie
        raise ValueError(f"{path} is not an executable (ELF type {e_type})")
    phoff, = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    for i in range(phnum):
        p_type, = struct.unpack_from("<I", data, phoff + i * phentsize)
        if p_type == 3:                         # PT_INTERP
            raise ValueError(f"{path} names a program interpreter, so it is not static")


def build_musl(crate: Path = CRATE, out: Path = OUT) -> Path:
    check_toolchain(musl=True, crate=crate)
    _cargo_build(["--target", TARGET], crate)
    built = crate / "target" / TARGET / "release" / "gmlx-entry"
    check_static_aarch64_elf(built)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    shutil.copyfile(built, tmp)
    os.chmod(tmp, 0o755)
    os.replace(tmp, out)
    return out


def build_native(crate: Path = CRATE) -> Path:
    check_toolchain(musl=False, crate=crate)
    _cargo_build([], crate)
    return crate / "target" / "release" / "gmlx-entry"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--native", action="store_true",
                    help="Build for this machine and print the binary's path.")
    a = ap.parse_args(argv)
    try:
        path = build_native() if a.native else build_musl()
    except ToolchainError as e:
        print(f"build_guest_entry: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"build_guest_entry: cargo failed with exit code {e.returncode}",
              file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"build_guest_entry: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
