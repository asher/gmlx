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
NOT_FOUND, CANNOT_RUN, LISTEN, ENDING = 127, 126, 125, 75


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


@pytest.fixture(autouse=True)
def session_dir(monkeypatch):
    """A session folder of this test's own. In a guest the entry uses
    /tmp/.gmlx-session, which tests running at once would share."""
    d = Path(tempfile.mkdtemp(prefix="gs-", dir="/tmp"))
    monkeypatch.setenv("GMLX_ENTRY_SESSION_DIR", str(d / "session"))
    yield d / "session"
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def short_dir():
    # macOS caps a Unix socket path at 104 bytes, so sockets live under /tmp.
    d = Path(tempfile.mkdtemp(prefix="ge-", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _free_port() -> int:
    """A port that was free a moment ago. The entry binds the port it is
    given, so the port must be closed here before the entry takes it, and
    another process can take it in between. The callers retry on exit 125,
    which makes that rare race harmless; launch never asks for port 0."""
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
    the client starts. The relay and the entry carry ``sock`` in their
    arguments, and the client does not."""
    for _ in range(3):
        port = _free_port()
        # The client must not inherit an ignored SIGINT from the test run,
        # or the Ctrl-C the test sends would not end it.
        client = subprocess.Popen(
            [entry, "--tcp", f"{port}={sock}", "--", "sh", "-c", "echo ready; exec sleep 30"],
            env=env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
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


def test_messages_escape_terminal_controls_in_the_image_path(entry, tmp_path):
    """The image's PATH and file names are the image's choice. An OSC 52
    clipboard write or a CSI cursor move in them prints as text."""
    evil = str(tmp_path / "b\x1b]52;c;ZXZpbA==\x07\x1b[2A")
    env = {**os.environ, "PATH": evil}
    done = _run(entry, "--check", "no-such-tool-xyz", env=env)
    assert done.returncode == NOT_FOUND
    assert "\x1b" not in done.stderr and "\x07" not in done.stderr
    assert "\\u{1b}]52;c;ZXZpbA==\\u{7}\\u{1b}[2A" in done.stderr
    os.makedirs(evil)
    tool = Path(evil) / "tool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o644)
    done = _run(entry, "--check", "tool", env=env)
    assert "no execute bit" in done.stderr and "\x1b" not in done.stderr
    tool.chmod(0o755)
    done = _run(entry, "--check", "tool", env=env)
    assert done.returncode == 0 and "\x1b" not in done.stdout


def test_run_execs_the_command_with_its_arguments_and_exit_code(entry):
    done = _run(entry, "--", "sh", "-c", 'echo "$0 $1"; exit 7', "zero", "one")
    assert done.returncode == 7
    assert done.stdout.strip() == "zero one"


def test_run_missing_command_exits_127(entry):
    done = _run(entry, "--", "no-such-tool-xyz")
    assert done.returncode == NOT_FOUND
    assert "Install it in the image" in done.stderr


def test_command_resolves_on_the_image_path_in_order(entry, tmp_path):
    tool = tmp_path / "bin" / "mytool"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho from-mytool\n")
    tool.chmod(0o755)
    # A second executable match later on PATH, so the test fails if the
    # search does not stop at the first one.
    later = _script(tmp_path / "later" / "mytool", "#!/bin/sh\necho from-later\n", 0o755)
    env = dict(os.environ, PATH=f"{tool.parent}:{later.parent}:/usr/bin:/bin")
    done = _run(entry, "--", "mytool", env=env)
    assert done.returncode == 0 and done.stdout.strip() == "from-mytool"
    done = _run(entry, "--check", "mytool", env=env)
    assert done.stdout.strip() == str(tool)
    env = dict(os.environ, PATH=f"{later.parent}:{tool.parent}:/usr/bin:/bin")
    assert _run(entry, "--", "mytool", env=env).stdout.strip() == "from-later"


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


@pytest.mark.parametrize("line, missing", [
    ("#!/nope/python3 -u", "/nope/python3"),
    ("#!/usr/bin/env missingtool", "missingtool"),
    ("#!/usr/bin/env -S -u HOME X=1 missingtool --flag", "missingtool"),
    # With no newline in the first 256 bytes, Linux runs the interpreter
    # with the argument cut short.
    ("#!/nope/x " + "a" * 300, "/nope/x"),
])
def test_a_missing_shebang_interpreter_exits_126(entry, tmp_path, line, missing):
    script = _script(tmp_path / "bin" / "start", f"{line}\necho ran\n", 0o755)
    env = dict(os.environ, PATH=f"{script.parent}:/usr/bin:/bin")
    message = f"gmlx-entry: {script} names {missing} in its #! line, which is not in the image."
    for args in (("--", "start"), ("--check", "start"), ("--", str(script)),
                 ("--check", str(script))):
        done = _run(entry, *args, env=env)
        assert done.returncode == CANNOT_RUN, (args, done.stderr)
        assert done.stderr.strip() == message, args
        assert done.stdout == "", args


def test_env_with_words_and_no_split_option_exits_126(entry, tmp_path):
    script = _script(tmp_path / "bin" / "start", "#!/usr/bin/env sh -e\necho ran\n", 0o755)
    env = dict(os.environ, PATH=f"{script.parent}:/usr/bin:/bin")
    message = (f'gmlx-entry: {script} has "sh -e" after env in its #! line, and env receives '
               "it as one command name. Write #!/usr/bin/env -S sh -e to pass it as separate "
               "words.")
    for args in (("--", "start"), ("--check", "start")):
        done = _run(entry, *args, env=env)
        assert done.returncode == CANNOT_RUN, (args, done.stderr)
        assert done.stderr.strip() == message, args


def test_a_cut_off_interpreter_name_is_left_to_exec(entry, tmp_path):
    # Linux refuses the file itself, so the check has nothing to add.
    script = _script(tmp_path / "bin" / "start", "#!/" + "a" * 300 + "\necho ran\n", 0o755)
    env = dict(os.environ, PATH=f"{script.parent}:/usr/bin:/bin")
    check = _run(entry, "--check", "start", env=env)
    assert check.returncode == 0 and check.stdout.strip() == str(script), check.stderr


@pytest.mark.parametrize("line", ["#!/bin/sh -e", "#! /usr/bin/env sh",
                                  "#!/usr/bin/env -S sh -e"])
def test_a_present_shebang_interpreter_runs(entry, tmp_path, line):
    script = _script(tmp_path / "bin" / "start", f"{line}\necho ran\n", 0o755)
    env = dict(os.environ, PATH=f"{script.parent}:/usr/bin:/bin")
    done = _run(entry, "--", "start", env=env)
    assert done.returncode == 0 and done.stdout.strip() == "ran", done.stderr
    check = _run(entry, "--check", "start", env=env)
    assert check.returncode == 0 and check.stdout.strip() == str(script), check.stderr


def test_a_windows_line_ending_in_the_shebang_exits_126(entry, tmp_path):
    script = _script(tmp_path / "bin" / "start", "#!/bin/sh\r\necho ran\r\n", 0o755)
    env = dict(os.environ, PATH=f"{script.parent}:/usr/bin:/bin")
    message = (f"gmlx-entry: {script} has a #! line that ends in a carriage return, from "
               "Windows line endings. Convert the file to Unix line endings.")
    for args in (("--", "start"), ("--check", "start")):
        done = _run(entry, *args, env=env)
        assert done.returncode == CANNOT_RUN, (args, done.stderr)
        assert done.stderr == message + "\n", args
        assert done.stdout == "", args


def test_shell_reports_a_bash_whose_interpreter_is_missing(entry, tmp_path):
    bash = _script(tmp_path / "bin" / "bash", "#!/nope/bash-real\n", 0o755)
    os.symlink(shutil.which("sh"), bash.parent / "sh")
    done = _run(entry, "--shell", "--", "-c", "echo x", env=dict(os.environ, PATH=str(bash.parent)))
    assert done.returncode == CANNOT_RUN
    assert done.stderr.strip() == (f"gmlx-entry: {bash} names /nope/bash-real in its #! line, "
                                   "which is not in the image.")


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
        relays = [pid for pid in _relay_pids(str(sock)) if pid != client.pid]
        assert relays
        assert all(os.getsid(pid) != os.getsid(client.pid) for pid in relays)
        os.killpg(client.pid, signal.SIGINT)         # passed on to the client
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


# The session and the copies that join it

def _start(entry, *args, **kw):
    """The entry in a session of its own, with pipes for its streams. The
    client must not inherit an ignored SIGINT from the test run."""
    return subprocess.Popen([entry, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=dict(os.environ), start_new_session=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
                            **kw)


def _send(proc, line: bytes = b"\n") -> None:
    proc.stdin.write(line)
    proc.stdin.flush()


def _gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _stop(*procs) -> None:
    for proc in procs:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)


def test_the_main_entry_waits_for_a_joined_copy_and_keeps_its_own_exit_code(entry):
    main = _start(entry, "--", "sh", "-c", "echo ready; read x; exit 3")
    copy = None
    try:
        assert main.stdout.readline() == b"ready\n"
        copy = _start(entry, "--join", "--", "sh", "-c", "echo joined; read x; exit 5")
        assert copy.stdout.readline() == b"joined\n"
        _send(main)
        assert main.stderr.readline() == (b"[launch] sh exited. The session stays open while "
                                          b"1 other copy runs.\n")
        assert main.poll() is None
        _send(copy)
        assert copy.wait(10) == 5
        assert main.wait(10) == 3
    finally:
        _stop(main, *([copy] if copy else []))


def test_a_copy_cannot_join_a_session_that_is_ending(entry, session_dir):
    import fcntl

    main = _start(entry, "--", "sh", "-c", "echo ready; read x")
    try:
        assert main.stdout.readline() == b"ready\n"
        # The main entry holds the lock exclusively while it ends.
        with open(session_dir / "copies.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            done = _run(entry, "--join", "--", "true")
        assert done.returncode == ENDING
        assert done.stderr == ("[launch] the session is ending, so this copy cannot join it. "
                               "Launch again once it has stopped.\n")
        _send(main)
        assert main.wait(10) == 0
        done = _run(entry, "--join", "--", "true")        # after the main entry exited
        assert done.returncode == ENDING and "is ending" in done.stderr
    finally:
        _stop(main)


def test_a_copy_without_a_session_folder_is_refused(entry, session_dir):
    done = _run(entry, "--join", "--", "true")
    assert done.returncode == ENDING
    assert f"its session folder {session_dir} is gone" in done.stderr


CLIENT = """
import os, signal, subprocess, sys
signal.signal(signal.SIGTERM, lambda *a: sys.exit(9))
signal.signal(signal.SIGINT, lambda *a: sys.exit(8))
child = subprocess.Popen(["sleep", "60"])
print(os.getpid(), os.getpgrp(), child.pid, flush=True)
signal.pause()
"""


@pytest.mark.parametrize("sig,code", [(signal.SIGTERM, 9), (signal.SIGINT, 8)])
def test_signals_reach_the_clients_process_group(entry, sig, code):
    main = _start(entry, "--", sys.executable, "-c", CLIENT)
    try:
        pid, group, grandchild = map(int, main.stdout.readline().split())
        assert group == pid and group != os.getpgid(main.pid)   # a group of its own
        main.send_signal(sig)
        assert main.wait(10) == code
        assert _gone(grandchild)                  # the whole group got the signal
    finally:
        _stop(main)


def test_the_lock_never_reaches_the_client(entry, session_dir):
    """A copy whose entry is killed leaves its client running, and that
    client holds no lock, so the session ends when the main client does."""
    import fcntl

    main = _start(entry, "--", "sh", "-c", "echo ready; read x; exit 6")
    copy = orphan = None
    try:
        assert main.stdout.readline() == b"ready\n"
        copy = _start(entry, "--join", "--", "sh", "-c", "echo $$; exec sleep 60")
        orphan = int(copy.stdout.readline())
        os.kill(copy.pid, signal.SIGKILL)
        copy.wait(10)
        with open(session_dir / "copies.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # nothing holds it now
        _send(main)
        assert main.wait(10) == 6
        assert main.stderr.read() == b""                        # it waited for nothing
    finally:
        if orphan:
            os.kill(orphan, signal.SIGKILL)
        _stop(main, *([copy] if copy else []))


def test_a_sigterm_while_waiting_ends_the_session_and_stops_the_copies(entry):
    main = _start(entry, "--", "sh", "-c", "echo ready; read x; exit 2")
    copy = _start(entry, "--join", "--", sys.executable, "-c", CLIENT)
    try:
        assert main.stdout.readline() == b"ready\n"
        copy.stdout.readline()
        _send(main)
        assert b"stays open" in main.stderr.readline()
        main.send_signal(signal.SIGTERM)
        assert main.wait(10) == 2
        assert copy.wait(10) == 9                 # the copy's client got SIGTERM
    finally:
        _stop(main, copy)


def test_a_second_ctrl_c_while_waiting_ends_the_session(entry):
    main = _start(entry, "--", "sh", "-c", "echo ready; read x; exit 0")
    copy = _start(entry, "--join", "--", "sh", "-c", "echo joined; read x")
    try:
        assert main.stdout.readline() == b"ready\n"
        assert copy.stdout.readline() == b"joined\n"
        _send(main)
        assert b"stays open" in main.stderr.readline()
        main.send_signal(signal.SIGINT)
        assert main.stderr.readline() == (b"[launch] Press Ctrl-C again to end the session, "
                                          b"which stops the other copy.\n")
        assert main.poll() is None
        main.send_signal(signal.SIGINT)
        assert main.wait(10) == 0
    finally:
        _stop(main, copy)


def _read_until(fd: int, text: bytes, timeout: float = 10.0) -> bytes:
    import select

    out = b""
    deadline = time.monotonic() + timeout
    while text not in out:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            pytest.fail(f"no {text!r} in {out!r}")
        out += os.read(fd, 4096)
    return out


def _foreground_group(pid: int) -> int:
    """The foreground process group of the terminal of ``pid``."""
    done = subprocess.run(["ps", "-o", "tpgid=", "-p", str(pid)], capture_output=True,
                          text=True, timeout=10)
    return int(done.stdout)


def test_the_client_gets_the_terminal_and_the_entry_takes_it_back(entry):
    login_tty = getattr(os, "login_tty", None)
    if login_tty is None or shutil.which("ps") is None:
        pytest.skip("needs os.login_tty and ps")
    master, slave = os.openpty()
    main = subprocess.Popen([entry, "--", "sh", "-c", "echo pid $$; read x; exit 4"],
                            stdin=slave, stdout=slave, stderr=slave, env=dict(os.environ),
                            preexec_fn=lambda: login_tty(slave))
    os.close(slave)
    copy = None
    try:
        client = int(_read_until(master, b"\n").split(b"pid ")[1].split()[0])
        assert _foreground_group(main.pid) == client != main.pid
        copy = _start(entry, "--join", "--", "sh", "-c", "echo joined; read x")
        assert copy.stdout.readline() == b"joined\n"
        os.write(master, b"\n")
        _read_until(master, b"stays open")
        assert _foreground_group(main.pid) == main.pid
        _send(copy)
        assert copy.wait(10) == 0
        assert main.wait(10) == 4
    finally:
        _stop(main, *([copy] if copy else []))
        os.close(master)


BACKGROUND = """
import os, sys
pid = os.fork()
if pid == 0:
    os.setpgid(0, 0)
    os.execv(sys.argv[1], sys.argv[1:])
os.setpgid(pid, pid)
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


def test_a_client_in_the_background_of_a_terminal_stays_in_the_entrys_group(entry):
    """A job-control shell starts the entry in a background group, so the
    client keeps the entry's group and gets no second SIGINT from it."""
    login_tty = getattr(os, "login_tty", None)
    if login_tty is None:
        pytest.skip("needs os.login_tty")
    client = CLIENT.replace("print(os.getpid()", "print(os.getppid(), os.getpid()")
    master, slave = os.openpty()
    shell = subprocess.Popen(
        [sys.executable, "-c", BACKGROUND, entry, "--", sys.executable, "-c", client],
        stdin=slave, stdout=slave, stderr=slave, env=dict(os.environ),
        preexec_fn=lambda: login_tty(slave))
    os.close(slave)
    main = None
    try:
        line = _read_until(master, b"\n").decode().split()
        main, pid, group = int(line[0]), int(line[1]), int(line[2])
        assert group == main == os.getpgid(main) != pid
        os.kill(main, signal.SIGINT)              # the client would exit 8
        os.kill(main, signal.SIGTERM)
        assert shell.wait(10) == 9
    finally:
        if main and not _gone(main, 0):
            os.killpg(main, signal.SIGKILL)
        _stop(shell)
        os.close(master)


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
