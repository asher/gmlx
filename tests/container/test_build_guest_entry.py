"""scripts/build_guest_entry.py: the toolchain check that runs before every
guest entry build, and the static-ELF check on its output. Fake cargo and
rustc scripts stand in for the toolchain, so nothing is compiled."""
from __future__ import annotations

import importlib.util
import os
import re
import struct
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "gmlx-entry"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_guest_entry")


def test_rust_version_matches_the_pinned_toolchain():
    manifest = tomllib.loads((CRATE / "Cargo.toml").read_text())
    assert manifest["package"]["rust-version"] == build.pinned_version()


def test_toolchain_file_names_the_musl_target():
    text = (CRATE / "rust-toolchain.toml").read_text()
    assert re.search(r'targets\s*=\s*\["aarch64-unknown-linux-musl"\]', text)


def _fake_tools(tmp_path, monkeypatch, *, cargo: str, rustc: str, sysroot: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cargo").write_text(f"#!/bin/sh\necho 'cargo {cargo} (abc 2025-10-21)'\n")
    (bin_dir / "rustc").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = --print ]; then echo {sysroot}; else echo "rustc {rustc} (x)"; fi\n')
    for tool in ("cargo", "rustc"):
        (bin_dir / tool).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")


def _sysroot(tmp_path, *, musl: bool) -> Path:
    root = tmp_path / "sysroot"
    (root / "lib" / "rustlib").mkdir(parents=True)
    if musl:
        (root / "lib" / "rustlib" / build.TARGET).mkdir()
    return root


def test_workflows_install_the_pinned_toolchain():
    version, target = build.pinned_version(CRATE), build.TARGET
    for name in ("test.yml", "release.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text()
        installs = re.findall(r"^[ \t]*(?:run: )?(rustup toolchain install[^\n]*)", text, re.M)
        assert installs, name
        for line in installs:
            assert line.split() == ["rustup", "toolchain", "install", version,
                                    "--profile", "minimal", "--target", target], line


def test_check_passes_with_the_pinned_version_and_target(tmp_path, monkeypatch):
    version = build.pinned_version()
    _fake_tools(tmp_path, monkeypatch, cargo=version, rustc=version,
                sysroot=_sysroot(tmp_path, musl=True))
    build.check_toolchain(musl=True)


@pytest.mark.parametrize("tool", ["cargo", "rustc"])
def test_check_refuses_another_version(tmp_path, monkeypatch, tool):
    version = build.pinned_version()
    versions = {"cargo": version, "rustc": version, tool: "1.0.0"}
    _fake_tools(tmp_path, monkeypatch, **versions, sysroot=_sysroot(tmp_path, musl=True))
    with pytest.raises(build.ToolchainError, match=f"{tool} is .*1.0.0.*pins {version}") as e:
        build.check_toolchain(musl=True)
    assert "rustup toolchain install" in str(e.value)


def test_check_refuses_a_sysroot_without_the_musl_target(tmp_path, monkeypatch):
    version = build.pinned_version()
    _fake_tools(tmp_path, monkeypatch, cargo=version, rustc=version,
                sysroot=_sysroot(tmp_path, musl=False))
    with pytest.raises(build.ToolchainError, match="no aarch64-unknown-linux-musl"):
        build.check_toolchain(musl=True)
    build.check_toolchain(musl=False)          # the native build needs no musl


def test_check_names_the_install_command_without_cargo(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(build.ToolchainError, match="cargo is not on PATH") as e:
        build.check_toolchain(musl=False)
    assert "brew install rustup" in str(e.value)


def test_check_turns_off_rustup_downloads(tmp_path, monkeypatch):
    version = build.pinned_version()
    _fake_tools(tmp_path, monkeypatch, cargo=version, rustc=version,
                sysroot=_sysroot(tmp_path, musl=True))
    (tmp_path / "bin" / "cargo").write_text(
        f'#!/bin/sh\n[ "$RUSTUP_AUTO_INSTALL" = 0 ] && echo "cargo {version} (x)"\n')
    build.check_toolchain(musl=False)


def _elf(*, machine=183, etype=2, interp=False) -> bytes:
    """A minimal 64-bit little-endian ELF header with one or two program
    headers."""
    phoff, phentsize = 64, 56
    types = [1] + ([3] if interp else [])
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4], header[5], header[6] = 2, 1, 1
    struct.pack_into("<HHI", header, 16, etype, machine, 1)
    struct.pack_into("<Q", header, 32, phoff)
    struct.pack_into("<HH", header, 54, phentsize, len(types))
    body = b"".join(struct.pack("<I", t) + bytes(phentsize - 4) for t in types)
    return bytes(header) + body


def test_elf_check_accepts_a_static_aarch64_executable(tmp_path):
    f = tmp_path / "ok"
    f.write_bytes(_elf())
    build.check_static_aarch64_elf(f)


@pytest.mark.parametrize("kwargs, match", [
    ({"machine": 62}, "not aarch64"),
    ({"interp": True}, "program interpreter"),
    ({"etype": 1}, "not an executable"),
])
def test_elf_check_refuses(tmp_path, kwargs, match):
    f = tmp_path / "bad"
    f.write_bytes(_elf(**kwargs))
    with pytest.raises(ValueError, match=match):
        build.check_static_aarch64_elf(f)


def test_elf_check_refuses_a_non_elf_file(tmp_path):
    f = tmp_path / "script"
    f.write_text("#!/bin/sh\n")
    with pytest.raises(ValueError, match="not a 64-bit"):
        build.check_static_aarch64_elf(f)


def test_dist_check_refuses_a_missing_or_non_executable_binary(tmp_path):
    import tarfile
    import zipfile
    dist = _load("check_guest_entry_dist")
    member = dist.MEMBER
    wheel = tmp_path / "gmlx-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        info = zipfile.ZipInfo(member)
        info.external_attr = 0o644 << 16
        z.writestr(info, _elf())
    with pytest.raises(SystemExit, match="no execute bit"):
        dist.check_wheel(wheel)
    sdist = tmp_path / "gmlx-0.tar.gz"
    with tarfile.open(sdist, "w:gz") as t:
        (tmp_path / "x").write_text("x")
        t.add(tmp_path / "x", arcname="gmlx-0/README.md")
    with pytest.raises(SystemExit, match="is missing"):
        dist.check_sdist(sdist)


def test_dist_check_refuses_the_symmetric_cases(tmp_path):
    import io
    import tarfile
    import zipfile
    dist = _load("check_guest_entry_dist")
    wheel = tmp_path / "gmlx-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("gmlx/__init__.py", "")
    with pytest.raises(SystemExit, match="is missing"):
        dist.check_wheel(wheel)
    sdist = tmp_path / "gmlx-0.tar.gz"
    data = _elf()
    with tarfile.open(sdist, "w:gz") as t:
        info = tarfile.TarInfo(f"gmlx-0/{dist.MEMBER}")
        info.size, info.mode = len(data), 0o644
        t.addfile(info, io.BytesIO(data))
    with pytest.raises(SystemExit, match="no execute bit"):
        dist.check_sdist(sdist)


def test_cargo_build_pins_the_target_folder_and_the_remaps(tmp_path, monkeypatch):
    seen = {}
    crate = tmp_path / "my src" / "gmlx-entry"
    home = tmp_path / "cargo home"
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("CARGO_HOME", str(home))
    monkeypatch.setenv("RUSTFLAGS", "-Cdebuginfo=2")
    monkeypatch.setattr(build.subprocess, "run",
                        lambda argv, cwd, env, check: seen.update(env=env))
    build._cargo_build([], crate)
    assert seen["env"]["CARGO_TARGET_DIR"] == str(crate / "target")
    assert "RUSTFLAGS" not in seen["env"]
    # Each flag stays whole even though both folder names hold a space.
    assert seen["env"]["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") == [
        f"--remap-path-prefix={crate}=/gmlx-entry", f"--remap-path-prefix={home}=/cargo"]


def test_gitignore_keeps_the_binary_out_of_git():
    text = (ROOT / ".gitignore").read_text()
    assert "/gmlx/container/guest/gmlx-entry" in text
    assert "/crates/gmlx-entry/target/" in text
    assert os.path.exists(CRATE / "Cargo.lock")


def test_package_data_covers_the_tracked_container_files():
    import subprocess
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        top = ""
    if not top or Path(top).resolve() != ROOT:
        pytest.skip("needs a git checkout, and an extracted sdist is not one")
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = manifest["tool"]["setuptools"]["package-data"]["gmlx.container"]
    tracked = subprocess.run(["git", "ls-files", "gmlx/container/files"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked
    for path in tracked:
        rel = str(Path(path).relative_to("gmlx/container"))
        assert any(Path(rel).match(p) for p in patterns), f"{path} is not package data"
