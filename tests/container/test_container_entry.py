"""The guest entry binary of `gmlx launch --container`, run for real.

The session fixture builds the crate for this machine through
scripts/build_guest_entry.py --native, so the build script's toolchain
check applies; a missing or unpinned toolchain fails the suite with the
install command. GMLX_ENTRY_BIN names a prebuilt binary instead, which the
arm64 Linux CI job uses for the release build. This file imports nothing
from gmlx and needs no conftest, so that job runs it with --noconftest.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOT_FOUND, CANNOT_RUN, LISTEN = 127, 126, 125


@pytest.fixture(scope="session")
def entry() -> str:
    given = os.environ.get("GMLX_ENTRY_BIN")
    if given:
        return given
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_guest_entry.py"),
                           "--native"], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.fail(f"cannot build the guest entry:\n{done.stderr.strip()}")
    return done.stdout.strip().splitlines()[-1]


@pytest.fixture
def short_dir():
    # macOS caps a Unix socket path at 104 bytes, so sockets live under /tmp.
    d = Path(tempfile.mkdtemp(prefix="ge-", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(entry, *args, env=None, **kw):
    return subprocess.run([entry, *args], capture_output=True, text=True,
                          env=env if env is not None else dict(os.environ),
                          timeout=30, **kw)


def _run_relay(entry, args_for, env):
    """Run the entry with a TCP listener on a free port. Another process can
    take the port between the probe and the entry's bind, which makes the
    entry exit 125, so a busy port is tried again with a new one."""
    for _ in range(3):
        port = _free_port()
        done = _run(entry, *args_for(port), env=env)
        if done.returncode != LISTEN:
            break
    return port, done


def _start_relay_client(entry, sock: Path, env) -> tuple:
    """Start the entry with a TCP listener in front of ``sock`` and a client
    that prints a line once it runs. The relay is bound and detached before
    the client starts, and the client has replaced the entry by then, so
    only the relay still carries ``sock`` in its arguments."""
    for _ in range(3):
        port = _free_port()
        client = subprocess.Popen(
            [entry, "--tcp", f"{port}={sock}", "--", "sh", "-c", "echo ready; exec sleep 30"],
            env=env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if client.stdout.readline() == b"ready\n":
            return client, port
        err = client.stderr.read().decode()
        client.wait(10)
        if client.returncode != LISTEN:
            pytest.fail(f"the entry exited {client.returncode}: {err}")
    pytest.fail(f"no free port after three tries: {err}")


def _relay_pids(marker: str) -> list:
    """The detached relay keeps the entry's argv, so a unique socket path in
    its arguments finds it."""
    done = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return [int(p) for p in done.stdout.split()]


def _kill_relays(marker: str) -> None:
    for pid in _relay_pids(marker):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _echo_unix(path: Path):
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(path))
    srv.listen(16)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                data = conn.recv(100)
                conn.sendall(b"echo " + data)

    threading.Thread(target=serve, daemon=True).start()
    return srv


def _connect_retry(port: int, timeout: float = 10.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_check_prints_the_resolved_path(entry):
    done = _run(entry, "--check", "sh")
    assert done.returncode == 0
    assert done.stdout.strip().endswith("/sh")


def test_check_missing_command_exits_127_with_a_message(entry):
    done = _run(entry, "--check", "no-such-tool-xyz")
    assert done.returncode == NOT_FOUND
    assert "no-such-tool-xyz is not on the image's PATH" in done.stderr
    assert done.stderr.count("\n") == 1                    # one line


def test_run_execs_the_command_with_its_arguments_and_exit_code(entry):
    done = _run(entry, "--", "sh", "-c", 'echo "$0 $1"; exit 7', "zero", "one")
    assert done.returncode == 7
    assert done.stdout.strip() == "zero one"


def test_run_missing_command_exits_127(entry):
    done = _run(entry, "--", "no-such-tool-xyz")
    assert done.returncode == NOT_FOUND
    assert "Install it in the image" in done.stderr


def test_command_resolves_on_the_image_path(entry, tmp_path):
    tool = tmp_path / "bin" / "mytool"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho from-mytool\n")
    tool.chmod(0o755)
    env = dict(os.environ, PATH=f"{tool.parent}:/usr/bin:/bin")
    done = _run(entry, "--", "mytool", env=env)
    assert done.returncode == 0 and done.stdout.strip() == "from-mytool"
    done = _run(entry, "--check", "mytool", env=env)
    assert done.stdout.strip() == str(tool)


def test_shell_falls_back_to_sh_and_passes_arguments(entry, tmp_path):
    only_sh = tmp_path / "bin"
    only_sh.mkdir()
    os.symlink(shutil.which("sh"), only_sh / "sh")
    env = dict(os.environ, PATH=str(only_sh))
    done = _run(entry, "--shell", "--", "-c", "echo shell-ran; exit 4", env=env)
    assert done.returncode == 4 and done.stdout.strip() == "shell-ran"


def test_shell_without_any_shell_exits_127(entry, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    done = _run(entry, "--shell", "--", env=dict(os.environ, PATH=str(empty)))
    assert done.returncode == NOT_FOUND
    assert "has no shell (bash or sh)" in done.stderr


def _script(path: Path, text: str, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def test_a_later_executable_match_wins_over_one_without_the_bit(entry, tmp_path):
    _script(tmp_path / "a" / "tool", "#!/bin/sh\necho from-a\n", 0o644)
    later = _script(tmp_path / "b" / "tool", "#!/bin/sh\necho from-b\n", 0o755)
    env = dict(os.environ, PATH=f"{tmp_path / 'a'}:{tmp_path / 'b'}:/usr/bin:/bin")
    done = _run(entry, "--", "tool", env=env)
    assert done.returncode == 0 and done.stdout.strip() == "from-b", done.stderr
    check = _run(entry, "--check", "tool", env=env)
    assert check.returncode == 0 and check.stdout.strip() == str(later), check.stderr


def test_a_match_without_the_execute_bit_exits_126(entry, tmp_path):
    plain = _script(tmp_path / "a" / "start.sh", "#!/bin/sh\necho ran\n", 0o644)
    env = dict(os.environ, PATH=f"{tmp_path / 'a'}:/usr/bin:/bin")
    message = f"{plain} has no execute bit. Run chmod 755 on it in the Containerfile."
    for args in (("--", "start.sh"), ("--check", "start.sh"), ("--", str(plain)),
                 ("--check", str(plain))):
        done = _run(entry, *args, env=env)
        assert done.returncode == CANNOT_RUN, (args, done.stderr)
        assert done.stderr.strip() == f"gmlx-entry: {message}", args
        assert done.stdout == "", args


def test_shell_skips_a_bash_without_the_execute_bit(entry, tmp_path):
    folder = tmp_path / "bin"
    _script(folder / "bash", "#!/bin/sh\necho wrong-shell\n", 0o644)
    os.symlink(shutil.which("sh"), folder / "sh")
    done = _run(entry, "--shell", "--", "-c", "echo sh-ran", env=dict(os.environ, PATH=str(folder)))
    assert done.returncode == 0 and done.stdout.strip() == "sh-ran", done.stderr


def test_busy_port_exits_125_with_a_message(entry, short_dir):
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    done = _run(entry, "--tcp", f"{port}={short_dir}/x.sock", "--", "true")
    held.close()
    assert done.returncode == LISTEN
    assert f"cannot listen on 127.0.0.1:{port}" in done.stderr


def test_tcp_listener_relays_to_a_unix_socket(entry, short_dir):
    sock = short_dir / "api.sock"
    srv = _echo_unix(sock)
    home = short_dir / "home"
    home.mkdir()
    try:
        port, done = _run_relay(entry, lambda port: ("--tcp", f"{port}={sock}", "--", "true"),
                                dict(os.environ, HOME=str(home)))
        assert done.returncode == 0, done.stderr     # the client ran and exited
        with _connect_retry(port) as c:              # the detached relay lives on
            c.sendall(b"ping")
            assert c.recv(100) == b"echo ping"
    finally:
        _kill_relays(str(sock))
        srv.close()


def test_unix_listener_relays_to_a_tcp_port(entry, short_dir):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def answer():
        conn, _ = srv.accept()
        with conn:
            conn.sendall(b"web " + conn.recv(10))

    threading.Thread(target=answer, daemon=True).start()
    web = short_dir / "nested" / "web.sock"          # the parent is created
    try:
        done = _run(entry, "--unix", f"{web}={port}", "--", "true",
                    env=dict(os.environ, HOME=str(short_dir)))
        assert done.returncode == 0, done.stderr
        c = socket.socket(socket.AF_UNIX)
        c.settimeout(5)
        c.connect(str(web))
        c.sendall(b"hi")
        assert c.recv(20) == b"web hi"
        c.close()
    finally:
        _kill_relays(str(web))
        srv.close()


def test_relay_has_its_own_session_and_survives_sigint(entry, short_dir):
    sock = short_dir / "api.sock"
    srv = _echo_unix(sock)
    client, port = _start_relay_client(entry, sock, dict(os.environ, HOME=str(short_dir)))
    try:
        with _connect_retry(port) as c:
            c.sendall(b"a")
            assert c.recv(10) == b"echo a"
        relays = _relay_pids(str(sock))
        assert relays and client.pid not in relays
        assert all(os.getsid(pid) != os.getsid(client.pid) for pid in relays)
        os.killpg(client.pid, signal.SIGINT)         # Ctrl-C to the client's group
        client.wait(10)
        with _connect_retry(port) as c:              # the server link survives
            c.sendall(b"b")
            assert c.recv(10) == b"echo b"
    finally:
        client.kill()
        _kill_relays(str(sock))
        srv.close()


def test_relay_log_is_truncated_per_session(entry, short_dir):
    log = short_dir / ".gmlx-entry.log"
    log.write_text("old session line\n" * 100)
    missing = short_dir / "missing.sock"
    try:
        port, done = _run_relay(entry, lambda port: ("--tcp", f"{port}={missing}", "--", "true"),
                                dict(os.environ, HOME=str(short_dir)))
        assert done.returncode == 0, done.stderr
        with _connect_retry(port) as c:
            assert c.recv(10) == b""                 # target down: closed at once
        deadline = time.monotonic() + 5
        while "cannot reach" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        text = log.read_text()
        assert "old session line" not in text
        assert f"port {port}: cannot reach {missing}" in text
    finally:
        _kill_relays(str(missing))


def test_relay_log_stops_at_one_mebibyte(entry, short_dir):
    missing = short_dir / "m.sock"
    log = short_dir / ".gmlx-entry.log"
    try:
        port, done = _run_relay(entry, lambda port: ("--tcp", f"{port}={missing}", "--", "true"),
                                dict(os.environ, HOME=str(short_dir)))
        assert done.returncode == 0, done.stderr
        line = len(f"port {port}: cannot reach {missing} (No such file or directory "
                   f"(os error 2))\n")
        for _ in range((1 << 20) // line + 200):
            with _connect_retry(port) as c:
                c.recv(1)
        time.sleep(0.5)
        assert (1 << 20) - line <= log.stat().st_size <= (1 << 20)
    finally:
        _kill_relays(str(missing))


def test_shell_without_listeners_starts_no_relay(entry, short_dir):
    done = _run(entry, "--shell", "--", "-c", "exit 0",
                env=dict(os.environ, HOME=str(short_dir)))
    assert done.returncode == 0
    assert not (short_dir / ".gmlx-entry.log").exists()   # the relay opens it


def test_bad_arguments_exit_2(entry):
    done = _run(entry, "--tcp", "notaport=/x", "--", "true")
    assert done.returncode == 2 and "usage:" in done.stderr


# The PATH under --clipboard

def test_clipboard_puts_the_stand_ins_first_after_resolving(entry, tmp_path):
    own = _script(tmp_path / "bin" / "xclip", '#!/bin/sh\necho "own xclip $PATH"\n', 0o755)
    # A stand-in folder with its own xclip: resolving CMD with it first
    # would run this one instead.
    stand_ins = _script(tmp_path / "stand-ins" / "xclip", "#!/bin/sh\necho stand-in\n",
                        0o755).parent
    env = dict(os.environ, PATH=f"{own.parent}:/usr/bin:/bin", GMLX_CLIP_BIN=str(stand_ins))
    done = _run(entry, "--clipboard", "--", "xclip", env=env)
    # The command resolved on the image's own PATH, and the client sees the
    # stand-ins first.
    assert done.stdout.strip() == f"own xclip {stand_ins}:{own.parent}:/usr/bin:/bin", done.stderr
    shell = _run(entry, "--clipboard", "--shell", "--", "-c", 'echo "$PATH"', env=env)
    assert shell.stdout.strip() == f"{stand_ins}:{own.parent}:/usr/bin:/bin"
    plain = _run(entry, "--clipboard", "--", "sh", "-c", 'echo "$PATH"',
                 env=dict(os.environ, PATH="/usr/bin:/bin"))
    assert plain.stdout.strip() == "/opt/gmlx/bin:/usr/bin:/bin"      # the default folder


def test_without_clipboard_the_image_path_stays(entry, tmp_path):
    env = dict(os.environ, PATH="/usr/bin:/bin")
    done = _run(entry, "--", "sh", "-c", 'echo "$PATH"', env=env)
    assert done.stdout.strip() == "/usr/bin:/bin"


# The clipboard stand-ins

def _stand_in(entry, folder: Path, name: str) -> str:
    link = folder / name
    link.symlink_to(entry)
    return str(link)


def _clip_server(path: Path, answer: bytes, requests: list):
    """A one-thread fake of the Mac clipboard socket."""
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(path))
    srv.listen(8)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                line = b""
                while not line.endswith(b"\n"):
                    chunk = conn.recv(1)
                    if not chunk:
                        break
                    line += chunk
                requests.append(line.decode())
                try:
                    conn.sendall(answer)
                except OSError:
                    pass                          # the stand-in stopped reading

    threading.Thread(target=serve, daemon=True).start()
    return srv


def _clip_run(link, *args, sock):
    return subprocess.run([link, *args], capture_output=True, timeout=30,
                          env=dict(os.environ, GMLX_CLIP_SOCK=str(sock)))


def test_stand_ins_read_an_image_through_their_links(entry, short_dir):
    sock = short_dir / "clip.sock"
    requests: list = []
    srv = _clip_server(sock, b"OK 8\n\x89PNG\r\n\x1a\n", requests)
    try:
        xclip = _stand_in(entry, short_dir, "xclip")
        wl = _stand_in(entry, short_dir, "wl-paste")
        done = _clip_run(xclip, "-selection", "clipboard", "-t", "image/png", "-o", sock=sock)
        assert done.returncode == 0 and done.stdout == b"\x89PNG\r\n\x1a\n"
        done = _clip_run(wl, "--type", "image/png", sock=sock)
        assert done.returncode == 0 and done.stdout == b"\x89PNG\r\n\x1a\n"
        done = _clip_run(xclip, "-selection", "clipboard", "-t", "TARGETS", "-o", sock=sock)
        assert done.returncode == 0
        assert requests == ["IMAGE image/png\n", "IMAGE image/png\n", "TYPES\n"]
    finally:
        srv.close()


def test_a_stand_in_piped_into_head_exits_quietly(entry, short_dir):
    sock = short_dir / "clip.sock"
    size = 4 << 20
    srv = _clip_server(sock, f"OK {size}\n".encode() + b"\0" * size, [])
    try:
        xclip = _stand_in(entry, short_dir, "xclip")
        done = subprocess.run(
            ["sh", "-c", f"'{xclip}' -selection clipboard -t image/png -o | head -c 1 >/dev/null"],
            capture_output=True, timeout=30, env=dict(os.environ, GMLX_CLIP_SOCK=str(sock)))
        assert done.returncode == 0 and done.stderr == b"", done.stderr
    finally:
        srv.close()


def test_stand_ins_pass_the_access_denied_message_on(entry, short_dir):
    sock = short_dir / "clip.sock"
    srv = _clip_server(sock, b"ERR macOS denies this app access to the clipboard. Allow it in "
                             b"System Settings, Privacy & Security, Paste from Other Apps.\n", [])
    try:
        done = _clip_run(_stand_in(entry, short_dir, "wl-paste"), "-t", "image/png", sock=sock)
        assert done.returncode == 1
        assert b"Paste from Other Apps" in done.stderr and done.stdout == b""
    finally:
        srv.close()


def test_stand_ins_without_the_socket_name_the_config_key(entry, short_dir):
    done = _clip_run(_stand_in(entry, short_dir, "xclip"), "-selection", "clipboard",
                     "-t", "image/png", "-o", sock=short_dir / "missing.sock")
    assert done.returncode == 1 and b"clipboard: images" in done.stderr


def test_stand_in_says_so_when_the_mac_side_stopped_answering(entry, short_dir):
    import socket
    sock = short_dir / "stale.sock"
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(sock))
    s.close()                                       # the file stays, nothing listens
    done = _clip_run(_stand_in(entry, short_dir, "xclip"), "-selection", "clipboard",
                     "-t", "image/png", "-o", sock=sock)
    assert done.returncode == 1 and b"does not answer" in done.stderr


def test_stand_ins_refuse_text_and_writes(entry, short_dir):
    sock = short_dir / "clip.sock"
    requests: list = []
    srv = _clip_server(sock, b"OK 0\n", requests)
    try:
        xsel = _stand_in(entry, short_dir, "xsel")
        xclip = _stand_in(entry, short_dir, "xclip")
        for link, args, message in [
            (xsel, ["--clipboard", "--output"], b"not text"),
            (xclip, ["-selection", "clipboard", "-o"], b"not text"),
            (xclip, ["-selection", "clipboard", "-i"], b"cannot write"),
        ]:
            done = _clip_run(link, *args, sock=sock)
            assert done.returncode == 1 and message in done.stderr, args
        done = subprocess.run([xclip, "-selection", "clipboard"], input=b"planted",
                              capture_output=True, timeout=30,
                              env=dict(os.environ, GMLX_CLIP_SOCK=str(sock)))
        assert done.returncode == 1 and b"cannot write" in done.stderr
        assert requests == []                  # nothing reached the Mac
    finally:
        srv.close()
