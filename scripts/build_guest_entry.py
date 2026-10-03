#!/usr/bin/env python3
"""Build the guest entry of `gmlx launch --container`.

The default build compiles crates/gmlx-entry for aarch64-unknown-linux-musl
in release mode and writes a static binary to gmlx/container/guest/gmlx-entry
with mode 0755. `--native` builds the same crate for this machine instead,
for the entry tests, and prints the binary's path.

Every build first checks the toolchain: `cargo` and `rustc`, run from the
crate folder, must be at least the rust-version that Cargo.toml states, and
the musl build also needs that target's standard library.
rust-toolchain.toml names the stable channel, so rustup builds with the
current release. Cargo fetches the one dependency, the `libc` crate, at the
version and checksum that Cargo.lock pins, so the first build needs the
network.

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
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "gmlx-entry"
TARGET = "aarch64-unknown-linux-musl"
OUT = ROOT / "gmlx" / "container" / "guest" / "gmlx-entry"
INSTALL_HINT = ("Install Rust as CONTRIBUTING.md describes:\n"
                "  brew install rustup\n"
                "  export PATH=\"$(brew --prefix rustup)/bin:$PATH\"\n"
                "  rustup toolchain install stable --profile minimal "
                f"--target {TARGET}")


class ToolchainError(RuntimeError):
    """The toolchain on PATH cannot build the crate."""


def min_version(crate: Path = CRATE) -> str:
    """The oldest compiler the crate builds with, the rust-version that
    Cargo.toml states."""
    manifest = tomllib.loads((crate / "Cargo.toml").read_text())
    version = manifest.get("package", {}).get("rust-version")
    if not isinstance(version, str) or _version(version) is None:
        raise ToolchainError(f"{crate / 'Cargo.toml'} states no rust-version")
    return version


def _version(text: str) -> tuple[int, int, int] | None:
    """The first x.y.z in ``text``, such as the version that
    ``rustc --version`` prints."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


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
    """Raise :class:`ToolchainError` unless cargo and rustc are at least the
    crate's rust-version and, for the musl build, the target's standard
    library exists."""
    floor = min_version(crate)
    for tool in ("cargo", "rustc"):
        try:
            line = _run_version(tool, crate)
        except ToolchainError as e:
            raise ToolchainError(f"{e}\n{INSTALL_HINT}")
        found = _version(line)
        if found is None or found < (_version(floor) or (0, 0, 0)):
            raise ToolchainError(f"{tool} is {line!r}, but the crate needs {floor} or "
                                 f"newer.\n{INSTALL_HINT}")
    if musl:
        done = subprocess.run(["rustc", "--print", "sysroot"], cwd=crate, env=_env(),
                              capture_output=True, text=True, timeout=60)
        lib = Path(done.stdout.strip()) / "lib" / "rustlib" / TARGET
        if done.returncode != 0 or not lib.is_dir():
            raise ToolchainError(f"rustc has no {TARGET} standard library "
                                 f"({lib} is missing).\n{INSTALL_HINT}")


def _rustflags(crate: Path) -> str:
    """Path remaps, so the binary never names a folder of the build machine.
    The flags are joined with 0x1f for CARGO_ENCODED_RUSTFLAGS, so a folder
    name with a space stays one flag."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    return "\x1f".join([f"--remap-path-prefix={crate}=/gmlx-entry",
                        f"--remap-path-prefix={cargo_home}=/cargo"])


def _cargo_build(args: list, crate: Path) -> None:
    env = _env()
    env["CARGO_ENCODED_RUSTFLAGS"] = _rustflags(crate)
    # The script reads the binary from the crate's own target folder. Cargo
    # ignores RUSTFLAGS when the encoded form is set, so it is removed to
    # keep the build free of flags from the caller.
    env["CARGO_TARGET_DIR"] = str(crate / "target")
    env.pop("RUSTFLAGS", None)
    subprocess.run(["cargo", "build", "--release", "--locked", *args],
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
