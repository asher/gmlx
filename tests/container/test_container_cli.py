"""gmlx/container/cli.py and images.py against a fake ``container`` command:
the wrapper's parsing, building and pulling images, the ``:base`` tag, digest
references, tag cleanup, the one-time command check and the rebuild hash."""

from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gmlx.config import LAUNCH_CLIENTS, LaunchCfg, LaunchClientCfg, LaunchContainerCfg
from gmlx.container import cli, ignore, images
from gmlx.container.state import FileLock

# The conftest makes the account home follow HOME in every test.
_ACCOUNT_HOME = cli.account_home

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


def _img(digest=D1, **kw):
    return {"digest": digest, **kw}


def _quiet(_line):
    pass


# The wrapper

def test_version_and_status(fake_container, tmp_path):
    fake_container.update(version="1.4.1")
    assert cli.version() == (1, 4, 1)
    assert cli.service() == cli.Service(True, cli.app_root())
    fake_container.update(app_root=str(tmp_path / "root"))
    assert cli.service() == cli.Service(True, tmp_path / "root")
    assert fake_container.calls("system", "status")[-1] == ["system", "status", "--format",
                                                            "json"]
    fake_container.update(running=False)
    assert cli.service() == cli.Service(False)


@pytest.mark.parametrize("out, rc, want", [
    ('{"status":"running","paths":{"appRoot":"relative/"}}', 0, cli.Service(True)),
    ('{"status":"running"}', 0, cli.Service(True)),
    ('{"status":"running","paths":{"appRoot":"/x/"}}', 1, cli.Service(False)),
    ('{"status":"unregistered"}', 1, cli.Service(False)),
    ("FIELD   VALUE\nstatus  running", 0, cli.Service(False)),
])
def test_service_reads_only_a_running_status(monkeypatch, out, rc, want):
    monkeypatch.setattr(cli, "_run", lambda args, **kw: subprocess.CompletedProcess(
        args, rc, out, ""))
    assert cli.service() == want


def test_kernel_installed_follows_the_app_root(monkeypatch, tmp_path):
    # Apple container ignores HOME, so a changed HOME moves nothing.
    import pwd

    monkeypatch.setenv("HOME", str(tmp_path))
    assert _ACCOUNT_HOME() == Path(pwd.getpwuid(os.getuid()).pw_dir) != tmp_path
    monkeypatch.setattr(cli, "account_home", lambda: tmp_path)
    assert cli.app_root() == tmp_path / "Library/Application Support/com.apple.container"
    assert not cli.kernel_installed()
    kernels = tmp_path / "root" / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "default.kernel-arm64").symlink_to(kernels / "vmlinux")
    assert not cli.kernel_installed(tmp_path / "root")      # a link to nothing
    (kernels / "vmlinux").write_bytes(b"kernel")
    assert cli.kernel_installed(tmp_path / "root")
    # `container system start` ignores the variable, so the start that
    # launch runs keeps its data in the default folder.
    monkeypatch.setenv("CONTAINER_APP_ROOT", str(tmp_path / "root"))
    assert not cli.kernel_installed()


def test_missing_binary_names_the_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cli.find() is None
    with pytest.raises(cli.Unavailable, match="brew install container") as e:
        cli.version()
    assert str(e.value).count("Apple container") == 1


def _program(folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "container"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_find_skips_the_current_folder_and_pin_keeps_the_first_find(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_pinned", None)
    monkeypatch.chdir(tmp_path)
    _program(tmp_path)                          # the current folder holds one
    monkeypatch.setenv("PATH", ":.:rel")
    assert cli.find() is None
    first = _program(tmp_path / "a")
    monkeypatch.setenv("PATH", f"{tmp_path / 'b'}::{tmp_path / 'a'}")
    assert cli.pin() == first
    # A client adds one earlier on PATH during the session.
    _program(tmp_path / "b")
    assert cli.find() == first
    monkeypatch.setenv("PATH", f"{tmp_path / 'b'}:{tmp_path / 'a'}:/x")
    assert cli.find() == str(tmp_path / "b" / "container")


def test_find_looks_in_no_folder_after_the_first_program(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_pinned", None)
    first = _program(tmp_path / "first")
    later = _program(tmp_path / "later")
    monkeypatch.setenv("PATH", os.pathsep.join(
        str(tmp_path / name) for name in ("first", "later", "mount")))
    # A later folder on a network mount that does not answer would make
    # each of these calls wait.
    seen = []
    isfile, access = os.path.isfile, os.access
    monkeypatch.setattr(os.path, "isfile", lambda p: seen.append(p) or isfile(p))
    monkeypatch.setattr(os, "access", lambda p, *a, **k: seen.append(p) or access(p, *a, **k))
    assert cli.find() == first
    assert {os.path.dirname(p) for p in seen} == {str(tmp_path / "first")}
    # The search for a newer program still gets every program on PATH.
    assert list(cli.on_path()) == [first, later]


def test_other_builds_runs_the_system_ps(monkeypatch):
    seen = []
    monkeypatch.setattr(images.subprocess, "run", lambda argv, *a, **k: seen.append(argv)
                        or subprocess.CompletedProcess(argv, 0, stdout=""))
    assert not images._other_builds()
    assert seen == [["/bin/ps", "-Ao", "command="]]


def test_image_info_reads_the_arm64_variant(fake_container):
    fake_container.update(images={"x:1": _img(
        arch=["linux/amd64", "linux/arm64"], entrypoint=["/bin/app"], cmd=["serve"],
        workdir="/app", created="2026-09-01T10:00:00.123Z")})
    info = cli.image_info("x:1")
    assert info.digest == D1 and info.arm64
    assert info.architectures == ["linux/amd64", "linux/arm64"]    # attestation skipped
    assert (info.entrypoint, info.cmd, info.workdir) == (["/bin/app"], ["serve"], "/app")
    assert info.created == datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    assert cli.image_info("missing:1") is None


def test_list_launch_containers_filters_by_label(fake_container):
    fake_container.update(containers=[
        {"name": "gmlx-pi-abc123", "labels": {"gmlx.launch": "1", "gmlx.launch.client": "pi"},
         "volumes": ["pg"], "image": "r@" + D1, "image_digest": D1},
        {"name": "other", "labels": {}}])
    found = cli.list_launch_containers()
    assert [c.name for c in found] == ["gmlx-pi-abc123"]
    assert found[0].volumes == ["pg"] and found[0].state == "running"


def test_memoized_queries_answer_once_until_a_change(fake_container):
    fake_container.update(images={"x:1": _img()}, containers=[{"name": "c1"}])
    with cli.memoized():
        assert [c.name for c in cli.containers()] == ["c1"]
        assert [c.name for c in cli.list_launch_containers()] == []
        assert cli.image_info("x:1").digest == D1
        assert cli.image_info("x:1").digest == D1
        assert cli.image_info("missing:1") is None
        assert cli.image_info("missing:1") is None     # a missing image is asked again
        cli.volume_list()
        cli.volume_list()
        assert len(fake_container.calls("ls")) == 1
        assert len(fake_container.calls("image", "inspect", "x:1")) == 1
        assert len(fake_container.calls("image", "inspect", "missing:1")) == 2
        assert len(fake_container.calls("volume", "list")) == 1
        cli.stop("c1")
        cli.containers()
        cli.tag("x:1", "x:2")
        cli.image_info("x:1")
        cli.volume_create("v", size="1G")
        cli.volume_list()
        assert len(fake_container.calls("ls")) == 2
        assert len(fake_container.calls("image", "inspect", "x:1")) == 2
        assert len(fake_container.calls("volume", "list")) == 2
        other = threading.Thread(target=cli.containers)   # another thread asks itself
        other.start()
        other.join()
        assert len(fake_container.calls("ls")) == 3
        cli.end_memo()
        cli.containers()
        assert len(fake_container.calls("ls")) == 4
    cli.containers()
    assert len(fake_container.calls("ls")) == 5


def test_volume_create_passes_the_label_and_size(fake_container):
    cli.volume_create("pg", size="32G")
    assert fake_container.calls("volume", "create") == [
        ["volume", "create", "--label", "gmlx.launch=1", "-s", "32G", "pg"]]
    vol = cli.volume_list()[0]
    assert vol.name == "pg" and vol.labels == {"gmlx.launch": "1"}


def test_only_remove_home_deletes_a_volume():
    """cli.py holds the one volume delete, images.py none, and only
    --remove-home calls it, after its question."""
    import re
    import subprocess
    wrapper = Path(cli.__file__).read_text()
    assert wrapper.count('"volume", "delete"') == 1
    assert "volume delete" not in Path(images.__file__).read_text()
    root = Path(cli.__file__).parents[1]
    hits = subprocess.run(["grep", "-rn", r"volume_delete(", str(root)], capture_output=True,
                          text=True).stdout.splitlines()
    callers = [h for h in hits if "def volume_delete" not in h]
    assert len(callers) == 1 and callers[0].startswith(str(root / "commands" / "launch_container.py"))
    text = Path(root / "commands" / "launch_container.py").read_text()
    start = text.index("def _remove_home(")
    body = text[start:text.index("\ndef ", start + 1)]
    assert "cli.volume_delete(" in body and not re.search(r"volume_delete\(", text.replace(body, ""))


def test_exec_argv():
    argv = cli.exec_argv("gmlx-pi-1", ["/opt/gmlx/gmlx-entry", "--shell"], tty=True, cwd="/w")
    assert argv[1:] == ["exec", "-i", "-t", "--cwd", "/w", "gmlx-pi-1",
                        "/opt/gmlx/gmlx-entry", "--shell"]


# Shipped images and :base

def test_shipped_build_tags_hash_and_base(fake_container):
    ready = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    build = fake_container.load()["builds"][0]
    tag = images.shipped_tag("pi", [])
    assert build["tags"] == [tag, "gmlx.invalid/launch-pi:base"]
    assert build["build_args"] == {"CLIENT": "pi", "EXTRA_PACKAGES": ""}
    assert not build["no_cache"] and not build["pull"]
    assert ready.action == "built" and ready.tag == tag
    assert ready.run_ref == f"gmlx.invalid/launch-pi@{ready.info.digest}"
    assert ["image", "tag", tag, ready.run_ref] in fake_container.log
    again = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert again.action == "found" and len(fake_container.load()["builds"]) == 1


@pytest.fixture
def no_other_builds(monkeypatch):
    monkeypatch.setattr(images, "_other_builds", lambda: False)


def test_a_builder_launch_started_is_stopped_after_the_build(fake_container, no_other_builds):
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop")
    assert not fake_container.load()["builder"]
    assert fake_container.load()["builder_args"] == [[]]


def test_a_stopped_builder_is_started_and_stopped_again(fake_container, no_other_builds):
    fake_container.update(builder=False)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop")


def test_a_running_builder_keeps_running_with_its_own_settings(fake_container, monkeypatch,
                                                               no_other_builds):
    fake_container.update(builder=True, builder_config={"cpus": 6, "memory": 8 << 30,
                                                        "ssh": False})
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert not fake_container.calls("builder", "stop")
    assert fake_container.load()["builder_args"] == [["--cpus", "6", "--memory", "8192M"]]


def test_a_launch_build_never_passes_ssh(monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    running = cli.Builder("running", cpus=4, memory_bytes=4 << 30, ssh=True)
    assert cli.builder_build_args(running) == ["--cpus", "4", "--memory", "4096M"]


@pytest.mark.parametrize("kind", ["shipped", "build"])
def test_a_builder_that_forwards_ssh_is_refused(fake_container, monkeypatch, no_other_builds,
                                               tmp_path, kind):
    """A RUN --mount=type=ssh step would reach every key in the Mac's agent,
    whatever ssh_agent says."""
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    fake_container.update(builder=True, builder_config={"cpus": 2, "memory": 2 << 30,
                                                        "ssh": True})
    plan = images.ImagePlan("shipped", "pi")
    if kind == "build":
        ctx = tmp_path / "ctx"
        ctx.mkdir()
        (ctx / "Containerfile").write_text("FROM debian\nRUN --mount=type=ssh true\n")
        plan = images.ImagePlan("build", "pi", containerfile=ctx / "Containerfile",
                                context=ctx)
    with pytest.raises(images.ImageError, match="before launch builds with: container builder stop$"):
        images.ensure_image(plan, say=_quiet)
    assert not fake_container.load().get("builds")
    assert not fake_container.calls("builder", "stop")


def test_the_builder_stays_while_another_launch_builds(fake_container, no_other_builds):
    other = FileLock(images.images_dir() / "builder.lock", shared=True)
    try:
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    finally:
        other.release()
    assert not fake_container.calls("builder", "stop")


def test_the_builder_stays_while_another_build_runs(fake_container, monkeypatch):
    monkeypatch.setattr(images, "_other_builds", lambda: True)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert not fake_container.calls("builder", "stop")


def test_other_builds_reads_the_process_list(monkeypatch):
    def ps(lines):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="\n".join(lines))
    monkeypatch.setattr(images.subprocess, "run", ps(["/usr/local/bin/container build -t x .",
                                                      "zsh"]))
    assert images._other_builds()
    monkeypatch.setattr(images.subprocess, "run", ps(["container builder start"]))
    assert images._other_builds()
    monkeypatch.setattr(images.subprocess, "run", ps([
        "/usr/local/bin/container-apiserver start", "container system status",
        "/usr/local/libexec/container/container-runtime-linux start --uuid buildkit"]))
    assert not images._other_builds()


def test_a_failed_builder_stop_only_warns(fake_container, no_other_builds):
    fake_container.update(fail_builder_stop=True)
    said = []
    ready = images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert ready.action == "built"
    warn = [line for line in said if "could not stop the image builder" in line]
    assert len(warn) == 1 and "container builder stop" in warn[0]


def test_a_failed_build_still_stops_the_builder_it_started(fake_container, no_other_builds):
    fake_container.update(fail_build=True)
    with pytest.raises(cli.ContainerError):
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop")


def test_a_closed_window_during_a_build_still_stops_the_builder(fake_container,
                                                               no_other_builds, monkeypatch):
    """The terminal's SIGHUP ends the build, and the shell's SIGHUP to its
    jobs comes while the clean-up asks for the builder."""
    import signal

    from gmlx.commands import launch_container as lc
    real_build, real_builder, real_popen = cli.build, cli.builder, cli.subprocess.Popen
    asked, groups = [], []

    def build(*args, **kw):
        real_build(*args, **kw)
        signal.raise_signal(signal.SIGHUP)

    def builder(**kw):
        asked.append(kw)
        if len(asked) == 2:
            signal.raise_signal(signal.SIGHUP)
        return real_builder(**kw)

    def popen(argv, **kw):
        groups.append((argv[1], kw.get("process_group")))
        return real_popen(argv, **kw)
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "builder", builder)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    with pytest.raises(lc._Signalled) as raised, lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert raised.value.signum == signal.SIGHUP
    assert fake_container.calls("builder", "stop") and not images._owed_path().exists()
    assert asked[1:] == [{"own_group": True}] * 2
    assert groups[-2:] == [("builder", 0), ("builder", 0)]


def test_a_double_ctrl_c_during_a_build_still_stops_the_builder(fake_container,
                                                                no_other_builds, monkeypatch):
    """The first Ctrl-C ends the build, and the second comes while the
    clean-up asks for the builder."""
    import signal

    from gmlx.commands import launch_container as lc
    real_build, real_builder = cli.build, cli.builder
    asked = []

    def build(*args, **kw):
        real_build(*args, **kw)
        signal.raise_signal(signal.SIGINT)

    def builder(**kw):
        asked.append(kw)
        if len(asked) == 2:
            signal.raise_signal(signal.SIGINT)
        return real_builder(**kw)
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "builder", builder)
    with pytest.raises(KeyboardInterrupt), lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop") and not images._owed_path().exists()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_third_signal_ends_the_clean_up_of_a_build(fake_container, no_other_builds,
                                                     monkeypatch):
    """A third signal stops the clean-up query, as when the container service
    does not answer it, and launch ends."""
    import signal

    from gmlx.commands import launch_container as lc
    real_build, real_builder = cli.build, cli.builder
    asked = []

    def build(*args, **kw):
        real_build(*args, **kw)
        signal.raise_signal(signal.SIGHUP)

    def builder(**kw):
        asked.append(kw)
        if len(asked) == 2:
            signal.raise_signal(signal.SIGHUP)
            signal.raise_signal(signal.SIGTERM)
        return real_builder(**kw)
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "builder", builder)
    with pytest.raises(lc._Signalled) as raised, lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert raised.value.signum == signal.SIGTERM
    assert len(asked) == 2 and images._owed_path().exists()
    assert not fake_container.calls("builder", "stop")


_LEFT_RUNNING = ("[launch] the image builder may still run and hold its memory. The next "
                 "launch stops it, or stop it now with: container builder stop")
_FINISHING = ("\n[launch] launch stops when its clean-up ends. Press Ctrl-C again to stop "
              "at once.\n")


def test_a_third_ctrl_c_during_the_builder_query_leaves_the_stop_to_the_next_launch(
        fake_container, no_other_builds, monkeypatch, capfd):
    """The first Ctrl-C ends the build, and the second and the third come
    while the clean-up asks for the builder. Launch ends with no further
    query, and its record lets the next launch stop the builder. The
    second Ctrl-C and the builder left running each get a line."""
    import signal

    from gmlx.commands import launch_container as lc
    fake_container.update(real_clock=True)
    real_build, real_builder = cli.build, cli.builder
    asked = []

    def send(signum):
        signal.raise_signal(signum)

    def build(*args, **kw):
        real_build(*args, **kw)
        send(signal.SIGINT)

    def builder(**kw):
        asked.append(kw)
        if len(asked) == 2:
            send(signal.SIGINT)            # ignored
            send(signal.SIGINT)            # ends the query
        return real_builder(**kw)
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "builder", builder)
    said = []
    capfd.readouterr()
    with pytest.raises(KeyboardInterrupt), lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert capfd.readouterr().err == _FINISHING
    assert [line for line in said if "image builder" in line] == [_LEFT_RUNNING]
    assert len(asked) == 2 and _stops(fake_container) == 0
    assert images._owed_path().exists()
    started = fake_container.load()["builder_started"]
    assert images._owes(images._read_date(images._owed_path()), started)
    assert images.builder_report()[1]                       # doctor warns
    assert images.builder_notice(say=_quiet) is None        # the next launch
    assert _stops(fake_container) == 1 and not images._owed_path().exists()


def test_a_third_ctrl_c_during_the_builder_stop_says_that_it_may_still_run(
        fake_container, no_other_builds, monkeypatch, capfd):
    """The first Ctrl-C ends the build, and the second and the third come
    while the clean-up stops the builder."""
    import signal

    from gmlx.commands import launch_container as lc
    fake_container.update(real_clock=True)
    real_build = cli.build

    def send(signum):
        signal.raise_signal(signum)

    def build(*args, **kw):
        real_build(*args, **kw)
        send(signal.SIGINT)

    def builder_stop(**_kw):
        send(signal.SIGINT)                # ignored
        send(signal.SIGINT)                # ends the stop
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "builder_stop", builder_stop)
    said = []
    capfd.readouterr()
    with pytest.raises(KeyboardInterrupt), lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert capfd.readouterr().err == _FINISHING
    assert [line for line in said if "image builder" in line] == [_LEFT_RUNNING]
    assert images._owed_path().exists()


@pytest.mark.parametrize("before", [True, False])
def test_a_signal_that_ends_the_clean_up_names_only_a_builder_that_launch_started(
        fake_container, no_other_builds, monkeypatch, before):
    """Launch owes no stop of a builder that ran before the build, such as
    one that you started, so it says nothing of it."""
    from gmlx.commands import launch_container as lc
    fake_container.update(real_clock=True)
    if before:
        fake_container.update(builder=True, builder_started="2026-09-28T09:00:00Z")
    real_build = cli.build

    def build(*args, **kw):
        real_build(*args, **kw)
        raise lc._Interrupted(True)        # as a third Ctrl-C during the build
    monkeypatch.setattr(cli, "build", build)
    said = []
    with pytest.raises(KeyboardInterrupt):
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    shown = [line for line in said if "image builder" in line]
    assert shown == ([] if before else [_LEFT_RUNNING]) and _stops(fake_container) == 0
    assert images._owed_path().exists() is not before


@pytest.mark.parametrize("signum", ["SIGTERM", "SIGHUP", "SIGINT"])
def test_a_first_signal_during_the_builder_query_still_stops_the_builder(
        fake_container, no_other_builds, monkeypatch, signum):
    """A first build ends by itself, and the first signal comes while the
    clean-up asks for the builder. The second signal would be ignored, so
    the builder clean-up of this launch still runs and stops the builder."""
    import signal

    from gmlx.commands import launch_container as lc
    signum = getattr(signal, signum)
    fake_container.update(real_clock=True)
    real_builder = cli.builder
    asked = []

    def builder(**kw):
        asked.append(kw)
        if len(asked) == 2:
            signal.raise_signal(signum)
        return real_builder(**kw)
    monkeypatch.setattr(cli, "builder", builder)
    raised = KeyboardInterrupt if signum == signal.SIGINT else lc._Signalled
    with pytest.raises(raised), lc._signals_raise():
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert len(asked) == 3 and _stops(fake_container) == 1
    assert not images._owed_path().exists()


@pytest.mark.parametrize("failed", [{2}, {2, 3}])
def test_a_failed_builder_query_after_a_first_build_keeps_the_owed_stop(
        fake_container, no_other_builds, monkeypatch, failed):
    """The query after a first build fails, as when it times out. The
    builder is stopped when the next query answers, in this launch or in
    the next one."""
    fake_container.update(real_clock=True)
    real_builder = cli.builder
    asked = []

    def builder(**kw):
        asked.append(kw)
        if len(asked) in failed:
            raise cli.Unavailable("`container builder status` gave no answer in 60 s.")
        return real_builder(**kw)
    monkeypatch.setattr(cli, "builder", builder)
    said = []
    assert images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append).action \
        == "built"
    assert len(asked) == 3
    warned = [line for line in said if "could not stop the image builder" in line]
    if failed == {2}:
        assert _stops(fake_container) == 1 and not images._owed_path().exists()
        assert not warned
        return
    assert _stops(fake_container) == 0 and len(warned) == 1
    started = fake_container.load()["builder_started"]
    assert images._owes(images._read_date(images._owed_path()), started)
    assert images.builder_report()[1]                       # doctor warns
    assert images.builder_notice(say=_quiet) is None        # the next launch
    assert _stops(fake_container) == 1 and not images._owed_path().exists()


def test_a_recorded_build_time_owes_only_a_builder_that_started_in_it(fake_container,
                                                                       no_other_builds):
    window = "2026-09-27T11:59:59Z 2026-09-27T12:00:05Z"
    images._write_date(images._owed_path(), window)
    fake_container.update(builder=True, builder_started="2026-09-28T09:00:00Z")  # your own
    assert images.builder_report()[1] is False
    assert images.builder_notice(say=_quiet) is not None    # a line, not a stop
    images._settle_builder(_quiet)
    assert _stops(fake_container) == 0 and not images._owed_path().exists()
    images._write_date(images._owed_path(), window)
    fake_container.update(builder_started="2026-09-27T12:00:05Z")
    assert images.builder_report()[1] is True
    images._settle_builder(_quiet)
    assert _stops(fake_container) == 1 and not images._owed_path().exists()


@pytest.mark.parametrize("record, started, owed", [
    ("2026-09-27T12:00:01Z", "2026-09-27T12:00:01Z", True),
    ("2026-09-27T12:00:01Z", "2026-09-27T12:00:02Z", False),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "2026-09-27T12:00:00Z", True),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "2026-09-27T12:00:02Z", True),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "2026-09-27T11:59:59Z", False),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "2026-09-27T12:00:03Z", False),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "2026-09-27T12:00:01", False),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", "770000000.5", False),
    ("2026-09-27T12:00:00Z 2026-09-27T12:00:02Z", None, False),
    (None, "2026-09-27T12:00:01Z", False),
])
def test_owes_matches_a_start_date_or_a_build_time(record, started, owed):
    assert images._owes(record, started) is owed


def test_utc_text_drops_the_fraction_of_a_second():
    assert images._utc_text(0.9) == "1970-01-01T00:00:00Z"
    assert images._utc_text(86401.0) == "1970-01-02T00:00:01Z"


def test_a_third_signal_during_the_build_s_last_moment_still_kills_it(tmp_path, monkeypatch):
    """The first signal ends the wait for the build's output, and the build
    gets a moment to end by itself. The second and the third signal come in
    that moment. The third ends it, and the build is still killed."""
    import select
    import signal

    from gmlx.commands import launch_container as lc
    program = tmp_path / "container"
    program.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
    program.chmod(0o755)
    monkeypatch.setattr(cli, "find", lambda: str(program))
    real_select, real_popen = select.select, cli.subprocess.Popen
    builds = []

    def send(signum):
        signal.raise_signal(signum)

    def first_select(*args):
        monkeypatch.setattr(cli.select, "select", real_select)
        send(signal.SIGTERM)
        return real_select(*args)

    class Build(real_popen):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            builds.append(self)

        def wait(self, timeout=None):
            if timeout == 1.0:
                send(signal.SIGTERM)       # ignored
                send(signal.SIGTERM)       # ends the moment
            return super().wait(timeout)
    monkeypatch.setattr(cli.select, "select", first_select)
    monkeypatch.setattr(cli.subprocess, "Popen", Build)
    try:
        with pytest.raises(lc._Signalled) as raised, lc._signals_raise():
            cli._run_watched(["build", "--file", "/ctx/Containerfile", "/ctx"])
        assert raised.value.signum == signal.SIGTERM
        assert len(builds) == 1 and builds[0].returncode == -signal.SIGKILL
    finally:
        for proc in builds:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_the_check_for_other_builds_has_a_process_group_of_its_own(monkeypatch):
    seen = []

    def run(argv, **kw):
        seen.append((kw.get("process_group"), kw.get("stdin")))
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(images.subprocess, "run", run)
    assert not images._other_builds()
    assert seen == [(0, subprocess.DEVNULL)]


_NPM_OFFLINE = ("npm error request to https://registry.npmjs.org/@anthropic-ai%2fclaude-code "
                "failed, reason: getaddrinfo EAI_AGAIN registry.npmjs.org")


def test_a_build_without_a_network_says_what_to_do(fake_container, capsys):
    fake_container.update(fail_build=_NPM_OFFLINE)
    with pytest.raises(cli.ContainerError) as e:
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    assert str(e.value) == cli.NO_NETWORK_HINT
    assert "VPN" in str(e.value) and "\n" not in str(e.value)
    err = capsys.readouterr().err
    assert "EAI_AGAIN" in err and "stderr tty: False" in err     # still on the terminal


def test_a_build_without_rosetta_turns_rosetta_off_for_the_builder(
        fake_container, no_other_builds, capsys, monkeypatch, tmp_path):
    """The builder uses Rosetta only for other architectures, so launch turns
    it off and names the restart, and does not advise installing Rosetta."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "ROSETTA_RUNTIME", tmp_path / "libRosettaRuntime")
    fake_container.update(fail_build="Error: internalError: \"failed to install rosetta\"")
    with pytest.raises(cli.ContainerError) as e:
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert str(e.value) == (
        "Apple's image builder cannot start, because the container service runs with Rosetta "
        "on for the builder, and Rosetta is not installed on this Mac. The images that launch "
        "builds do not need Rosetta. Launch set rosetta = false under [build] in "
        "~/.config/container/config.toml. The service reads that file only when it starts, "
        "so stop it with: container system stop. That stops every running container. Then "
        "launch again.")
    assert "softwareupdate" not in str(e.value) and "--rebuild" not in str(e.value)
    assert (tmp_path / ".config" / "container" / "config.toml").read_text() == (
        "[build]\nrosetta = false\n")
    assert "failed to install rosetta" in capsys.readouterr().err


@pytest.mark.parametrize("before, after", [
    (None, "[build]\nrosetta = false\n"),
    ("", "[build]\nrosetta = false\n"),
    ("[container]\ncpus = 4\n", "[container]\ncpus = 4\n\n[build]\nrosetta = false\n"),
    ("[container]\ncpus = 4", "[container]\ncpus = 4\n\n[build]\nrosetta = false\n"),
    ("[build]  # mine\ncpus = 4\n\n[dns]\n",
     "[build]  # mine\nrosetta = false\ncpus = 4\n\n[dns]\n")])
def test_rosetta_off_edits_only_what_it_needs(monkeypatch, tmp_path, before, after):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".config" / "container" / "config.toml"
    if before is not None:
        path.parent.mkdir(parents=True)
        path.write_text(before)
        path.chmod(0o640)
    assert cli.builder_rosetta_off() == (True, None)
    assert path.read_text() == after
    if before is not None:
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert cli.builder_rosetta_off() == (False, None)          # already off


@pytest.mark.parametrize("text, problem", [
    ("[build]\nrosetta = true\n", "it sets rosetta = true under [build]"),
    ("[build\n", "it is not valid TOML"),
    ("build = { cpus = 4 }\n", "it sets the build settings in a form that launch does not edit"),
    ("build.cpus = 4\n", "it sets the build settings in a form that launch does not edit")])
def test_rosetta_off_leaves_a_file_it_cannot_change_as_it_is(monkeypatch, tmp_path, text,
                                                             problem):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".config" / "container" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    assert cli.builder_rosetta_off() == (False, problem)
    assert path.read_text() == text
    assert f"Launch did not change ~/.config/container/config.toml, because {problem}. Set " \
           f"rosetta = false under [build] there." in cli.rosetta_refusal()


def test_a_failed_command_names_the_service_log_once(fake_container):
    fake_container.update(inspect_error="internalError: the store is locked")
    with pytest.raises(cli.CommandFailed) as e:
        cli.image_info("debian:12")
    assert str(e.value) == ("`container image inspect debian:12` failed (exit 1): Error: "
                            "internalError: the store is locked")
    assert cli.report(e.value) == (
        f"{e.value}. If that does not name the cause, read the service log with: container "
        "system logs")
    assert cli.report(RuntimeError("plain.")) == "plain."


def test_a_mac_with_rosetta_keeps_the_build_s_own_message(fake_container, capsys,
                                                         monkeypatch, tmp_path):
    """A step of the build can print the words, such as a package whose name
    holds "rosetta"."""
    runtime = tmp_path / "libRosettaRuntime"
    runtime.write_bytes(b"")
    monkeypatch.setattr(cli, "ROSETTA_RUNTIME", runtime)
    fake_container.update(fail_build="ERROR: Failed to install rosetta-sdk: no matching "
                                     "distribution")
    with pytest.raises(cli.BuildFailed, match=r"^`container build --file /ctx/Containerfile` "
                                              r"failed \(exit 1\)\.$"):
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    assert "rosetta-sdk" in capsys.readouterr().err


def test_other_build_failures_keep_their_message(fake_container, capsys):
    fake_container.update(fail_build="[ERROR] Could not resolve dependencies for project")
    with pytest.raises(cli.ContainerError, match=r"`container build --file /ctx/Containerfile` "
                                                 r"failed \(exit 1\)\.$"):
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    assert "Could not resolve dependencies" in capsys.readouterr().err


def test_a_failed_build_names_its_file_after_the_builder_options(fake_container):
    fake_container.update(fail_build="exit code: 1")
    with pytest.raises(cli.BuildFailed, match=r"^`container build --file /ctx/Containerfile` "
                                              r"failed \(exit 1\)\.$"):
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"],
                  builder_args=["--cpus", "2", "--memory", "2048M"])


@pytest.mark.parametrize("packages,next_step", [
    (["nosuch"], "When it names a package from packages, fix that entry. Otherwise launch "
                 "again with --rebuild."),
    ([], "Launch again with --rebuild.")])
def test_a_failed_shipped_build_names_the_next_step(fake_container, no_other_builds, capsys,
                                                    packages, next_step):
    fake_container.update(fail_build="E: Unable to locate package nosuch")
    with pytest.raises(images.ImageError) as e:
        images.ensure_image(images.ImagePlan("shipped", "pi", packages=packages), say=_quiet)
    first, link = str(e.value).split("\n")
    assert first == ("the build of the pi image failed (exit 1). The build output above shows "
                     f"the failing step. {next_step}")
    assert link == f"See {images.BUILD_FAILED_URL}"
    anchor = images.BUILD_FAILED_URL.rsplit("#", 1)[1]
    headings = (Path(__file__).parents[2] / "docs" / "troubleshooting.md").read_text()
    assert f"### {anchor.replace('-', ' ').capitalize()}\n" in headings
    assert "Unable to locate package" in capsys.readouterr().err


def test_a_build_on_a_terminal_draws_on_a_terminal(fake_container, monkeypatch):
    """container build draws its progress only when standard error is a
    terminal, so launch gives it a pseudo-terminal while it reads it."""
    import io

    class Tty(io.TextIOWrapper):
        def isatty(self):
            return True
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stderr", Tty(raw, encoding="utf-8"))
    fake_container.update(fail_build="getaddrinfo ENOTFOUND registry.npmjs.org")
    with pytest.raises(cli.ContainerError, match="VPN"):
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    assert b"stderr tty: True" in raw.getvalue()


def test_a_build_follows_a_terminal_resize(fake_container, monkeypatch):
    """A resize reaches the build's pseudo-terminal, and the build gets a
    SIGWINCH to draw again at the new size."""
    import fcntl
    import pty
    import signal
    import struct
    import termios

    term, term_slave = pty.openpty()
    fcntl.ioctl(term, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    stderr = os.fdopen(term_slave, "w")
    monkeypatch.setattr(sys, "stderr", stderr)
    fake_container.update(build_waits_for_resize=True)
    before = signal.getsignal(signal.SIGWINCH)
    seen = bytearray()

    def terminal():
        # Read what the build draws, and resize once it waits.
        resized = False
        while True:
            try:
                chunk = os.read(term, 4096)
            except OSError:
                return
            if not chunk:
                return
            seen.extend(chunk)
            if not resized and b"waiting for a resize" in seen:
                resized = True
                fcntl.ioctl(term, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 132, 0, 0))
                os.kill(os.getpid(), signal.SIGWINCH)
    reader = threading.Thread(target=terminal, daemon=True)
    reader.start()
    try:
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    finally:
        stderr.close()
        reader.join(10)
        os.close(term)
    assert b"got SIGWINCH at 132x50" in seen
    assert signal.getsignal(signal.SIGWINCH) == before


def test_a_build_off_the_main_thread_skips_resizes(fake_container, monkeypatch):
    import io

    class Tty(io.TextIOWrapper):
        def isatty(self):
            return True
    monkeypatch.setattr(sys, "stderr", Tty(io.BytesIO(), encoding="utf-8"))
    errors = []

    def build():
        try:
            cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
        except BaseException as e:           # noqa: BLE001 (reported below)
            errors.append(e)
    t = threading.Thread(target=build)
    t.start()
    t.join(30)
    assert not t.is_alive() and errors == [] and fake_container.calls("build")


def test_builder_notice(fake_container, no_other_builds):
    assert images.builder_notice() is None                  # no builder
    fake_container.update(builder=False)
    assert images.builder_notice() is None
    fake_container.update(builder=True, builder_config={"cpus": 2, "memory": 4 << 30,
                                                        "ssh": False})
    notice = images.builder_notice()
    assert "holds 4 GB" in notice and notice.endswith("Stop it with: container builder stop")
    held = FileLock(images.images_dir() / "builder.lock", shared=True)
    try:
        assert images.builder_notice() is None              # a launch is building
    finally:
        held.release()
    fake_container.update(running=False)
    fake_container.update(inspect_error="x")
    os.environ["PATH"], saved = "/nonexistent", os.environ["PATH"]
    try:
        assert images.builder_notice() is None              # never raises
    finally:
        os.environ["PATH"] = saved


def _stops(fake_container) -> int:
    return len(fake_container.calls("builder", "stop"))


def test_overlapping_builds_leave_no_builder_running(fake_container, no_other_builds):
    """Launch A starts the builder while launch B holds the lock, so A
    cannot stop it. B stops it when it settles, because the recorded start
    date still names the builder that runs."""
    other = FileLock(images.images_dir() / "builder.lock", shared=True)
    try:
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
        assert _stops(fake_container) == 0
    finally:
        other.release()
    images._settle_builder(_quiet)
    assert _stops(fake_container) == 1 and not fake_container.load()["builder"]
    assert not images._owed_path().exists()


def test_a_builder_you_started_later_is_never_stopped(fake_container, no_other_builds):
    other = FileLock(images.images_dir() / "builder.lock", shared=True)
    try:
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    finally:
        other.release()
    state = fake_container.load()
    state.update(builder=True, builder_started="2026-09-28T09:00:00Z")   # your own start
    fake_container.save(state)
    images._settle_builder(_quiet)
    assert _stops(fake_container) == 0 and fake_container.load()["builder"]
    assert not images._owed_path().exists()


def test_a_base_and_a_user_build_start_and_stop_the_builder_once(fake_container,
                                                                 no_other_builds, tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM gmlx.invalid/launch-pi:base\n")
    plan = images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), LaunchContainerCfg())
    images.ensure_image(plan, say=_quiet)
    state = fake_container.load()
    assert len(state["builds"]) == 2
    assert state["builder_starts"] == 1 and _stops(fake_container) == 1


def test_the_idle_builder_notice_comes_once_per_start(fake_container, no_other_builds):
    fake_container.update(builder=True, builder_started="2026-09-28T09:00:00Z")
    assert images.builder_notice() is not None
    assert images.builder_notice() is None
    line, owed = images.builder_report()
    assert not owed and line.endswith("container builder stop")
    fake_container.update(builder_started="2026-09-28T10:00:00Z")    # started again
    assert images.builder_notice() is not None
    assert _stops(fake_container) == 0


def test_an_owed_stop_is_settled_by_the_next_launch(fake_container, no_other_builds,
                                                   monkeypatch):
    """A launch killed after its build leaves the stop owed. The next launch
    stops that builder instead of reporting it, and doctor reports it as
    owed until then."""
    settle = images._settle_builder
    monkeypatch.setattr(images, "_settle_builder", lambda say: None)   # killed first
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    monkeypatch.setattr(images, "_settle_builder", settle)
    line, owed = images.builder_report()
    assert owed and "container builder stop" in line
    assert images.builder_notice() is None
    assert _stops(fake_container) == 1 and not fake_container.load()["builder"]


@pytest.mark.parametrize("env_set, builder_env", [
    ({"NO_COLOR": "1"}, []),
    ({}, ["NO_COLOR=true"]),
    ({"BUILDKIT_COLORS": "run=green"}, ["BUILDKIT_COLORS=error=red"]),
])
def test_a_running_builder_keeps_its_colour_settings(fake_container, no_other_builds,
                                                    monkeypatch, env_set, builder_env):
    for name in ("NO_COLOR", "BUILDKIT_COLORS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env_set.items():
        monkeypatch.setenv(name, value)
    fake_container.update(builder=True, builder_env=builder_env)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    state = fake_container.load()
    assert state.get("builder_recreated", 0) == 0
    assert state["builder_env"] == builder_env


def test_a_build_waits_while_the_builder_stops(fake_container, no_other_builds, monkeypatch):
    # images.time is the time module, so subprocess sees the patch too: its
    # wait for a child that closed its pipes but has not exited yet sleeps
    # in short steps under load. Only the waits of images count here.
    import sys
    waits = []
    real = images.time.sleep

    def sleep(seconds):
        if sys._getframe(1).f_globals.get("__name__") == images.__name__:
            waits.append(seconds)
        else:
            real(seconds)
    monkeypatch.setattr(images.time, "sleep", sleep)
    fake_container.update(builder=False, builder_stopping=2)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert waits == [0.5, 0.5] and len(fake_container.load()["builds"]) == 1


@pytest.mark.parametrize("line, found", [
    ("container --debug build -t x .", True),
    ("/usr/local/bin/container --debug --help build", True),
    ("container -d builder start", True),
    ("container --debug run --rm build", False),
    ("container-apiserver start", False),
])
def test_the_build_pattern_accepts_options_first(line, found):
    assert bool(images._BUILD_COMMAND.search(line)) is found


def test_old_list_records_are_migrated_and_cleaned(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    path = images._records_path()
    records = _records()
    records[first.run_ref] = ["pi"]
    path.write_text(json.dumps(records))
    assert _records()[first.run_ref] == {"clients": ["pi"], "pids": []}
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert first.run_ref not in fake_container.load()["images"]
    assert first.run_ref not in _records()


def test_the_gone_image_sweep_covers_only_the_cleaned_repository(fake_container):
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    elsewhere = "gmlx.invalid/launch-omp@" + D2        # not in the store

    def plant(records):
        records[elsewhere] = {"clients": ["omp"], "pids": []}
    images._update_records(plant)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert elsewhere in _records()


def test_packages_change_the_hash_and_reach_the_build(fake_container):
    assert images.shipped_tag("pi", []) != images.shipped_tag("pi", ["make"])
    assert images.shipped_tag("pi", ["make", "jq"]) == images.shipped_tag("pi", ["jq", "make"])
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make", "jq"]), say=_quiet)
    assert fake_container.load()["builds"][0]["build_args"]["EXTRA_PACKAGES"] == "jq make"


def test_base_is_retagged_when_it_names_another_image(fake_container):
    tag = images.shipped_tag("pi", [])
    fake_container.update(images={tag: _img(D1), "gmlx.invalid/launch-pi:base": _img(D2)})
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert ["image", "tag", tag, "gmlx.invalid/launch-pi:base"] in fake_container.log
    assert "builds" not in fake_container.load()


def test_rebuild_passes_no_cache_and_pull(fake_container):
    images.ensure_image(images.ImagePlan("shipped", "pi"), rebuild=True, say=_quiet)
    build = fake_container.load()["builds"][0]
    assert build["no_cache"] and build["pull"]


def test_old_tags_go_but_base_other_repos_and_in_use_stay(fake_container):
    old, in_use = "gmlx.invalid/launch-pi:0000000000000000", "gmlx.invalid/launch-pi:1111111111111111"
    fake_container.update(
        images={old: _img("sha256:" + "a" * 64), in_use: _img("sha256:" + "b" * 64),
                "gmlx.invalid/launch-pi-build:2222222222222222": _img("sha256:" + "c" * 64),
                "gmlx.invalid/launch-omp:0000000000000000": _img("sha256:" + "d" * 64)},
        containers=[{"name": "gmlx-pi-1", "labels": {"gmlx.launch": "1"},
                     "image": in_use, "image_digest": "sha256:" + "b" * 64}])
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    store = fake_container.load()["images"]
    assert old not in store
    assert in_use in store and "gmlx.invalid/launch-pi:base" in store
    assert "gmlx.invalid/launch-pi-build:2222222222222222" in store
    assert "gmlx.invalid/launch-omp:0000000000000000" in store


def test_cleanup_deletes_only_recorded_digest_references(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    stray = "gmlx.invalid/launch-pi@sha256:" + "e" * 64
    state = fake_container.load()
    state["images"][stray] = _img("sha256:" + "e" * 64)
    fake_container.save(state)
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    store = fake_container.load()["images"]
    assert first.run_ref not in store            # recorded by launch, replaced
    assert stray in store                         # never recorded, kept


def _set_pin_pid(run_ref, pid):
    def edit(records):
        records[run_ref]["pids"] = [pid]
    images._update_records(edit)


def test_cleanup_keeps_a_reference_another_running_launch_pinned(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    # Another launch pinned it and has not reached `container run` yet.
    other = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                             stdin=subprocess.PIPE)
    try:
        _set_pin_pid(first.run_ref, other.pid)
        images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
        assert first.run_ref in fake_container.load()["images"]
    finally:
        other.communicate(b"")
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert first.run_ref not in fake_container.load()["images"]


def test_cleanup_keeps_a_digest_reference_a_running_container_uses(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    fake_container.update(containers=[{
        "name": "gmlx-pi-aaaaaa", "labels": {"gmlx.launch": "1"},
        "image": first.run_ref, "image_digest": ""}])
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert first.run_ref in fake_container.load()["images"]


def test_a_refused_delete_stays_recorded(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    fake_container.update(refuse_delete=[first.run_ref])
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert first.run_ref in fake_container.load()["images"]
    assert first.run_ref in images._update_records(lambda r: None)


def test_base_refs_in_accepts_known_bases_and_refuses_the_rest():
    text = "FROM gmlx.invalid/launch-claude-code:base\nCOPY --from=gmlx.invalid/launch-pi:base /x /x\n"
    assert images.base_refs_in(text) == ["claude-code", "pi"]
    for bad in ("FROM gmlx.invalid/launch-pi:0123456789abcdef\n",
                "FROM gmlx.invalid/launch-nope:base\n", "FROM gmlx.invalid/other\n"):
        with pytest.raises(images.ImageError, match="Name the base of a client"):
            images.base_refs_in(bad)


# build: images

def _user_build(tmp_path, text="FROM gmlx.invalid/launch-pi:base\nRUN true\n", packages=()):
    ctx = tmp_path / "ctx"
    ctx.mkdir(exist_ok=True)
    (ctx / "Containerfile").write_text(text)
    cfg = LaunchContainerCfg(clients={"pi": LaunchClientCfg(build=str(ctx),
                                                           packages=list(packages))})
    return ctx, images.resolve_image("pi", cfg.for_client("pi"), cfg)


def test_user_build_builds_the_base_first(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path)
    assert plan.kind == "build" and plan.bases == ["pi"]
    ready = images.ensure_image(plan, say=_quiet)
    builds = fake_container.load()["builds"]
    assert builds[0]["tags"][1] == "gmlx.invalid/launch-pi:base"
    assert builds[1]["tags"][0].startswith("gmlx.invalid/launch-pi-build:")
    assert builds[1]["context"] == str(ctx)
    assert ready.kind == "build" and ready.run_ref.startswith("gmlx.invalid/launch-pi-build@")


def test_a_failed_user_build_names_the_containerfile(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path)
    images.ensure_image(plan, say=_quiet)
    # The base is found, and the builder runs, so the build passes its options.
    fake_container.update(builder=True)
    (ctx / "Containerfile").write_text("FROM gmlx.invalid/launch-pi:base\nRUN false\n")
    fake_container.update(fail_build="process did not complete successfully: exit code: 1")
    with pytest.raises(images.ImageError) as e:
        images.ensure_image(plan, say=_quiet)
    first, link = str(e.value).split("\n")
    assert first == (f"the build of {ctx / 'Containerfile'} failed (exit 1). The build output "
                     "above shows the failing step. Fix that step, or launch again with "
                     "--rebuild.")
    assert link == f"See {images.BUILD_FAILED_URL}"


def test_user_hash_follows_the_base_digest(fake_container, tmp_path):
    _, plan = _user_build(tmp_path)
    one = images.build_hash(plan, {"pi": D1}, _quiet)
    assert one != images.build_hash(plan, {"pi": D2}, _quiet)


def test_rebuild_with_a_base_pulls_only_the_base(fake_container, tmp_path):
    _, plan = _user_build(tmp_path)
    images.ensure_image(plan, rebuild=True, say=_quiet)
    base, user = fake_container.load()["builds"]
    assert base["no_cache"] and base["pull"]
    assert user["no_cache"] and not user["pull"]


def test_rebuild_without_a_base_pulls(fake_container, tmp_path):
    _, plan = _user_build(tmp_path, text="FROM debian:bookworm-slim\n")
    images.ensure_image(plan, rebuild=True, say=_quiet)
    (user,) = fake_container.load()["builds"]
    assert user["no_cache"] and user["pull"]


def test_ignored_files_do_not_change_the_hash(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    (ctx / ".dockerignore").write_text("node_modules\n")
    (ctx / "node_modules").mkdir()
    (ctx / "node_modules" / "a.js").write_text("1")
    (ctx / "app.js").write_text("1")
    (ctx / ".git").mkdir()
    (ctx / ".git" / "HEAD").write_text("ref")
    before = images.build_hash(plan, {}, _quiet)
    (ctx / "node_modules" / "a.js").write_text("22")
    (ctx / ".git" / "HEAD").write_text("other")
    assert images.build_hash(plan, {}, _quiet) == before
    (ctx / "app.js").write_text("22")
    assert images.build_hash(plan, {}, _quiet) != before


def test_unsupported_ignore_pattern_counts_every_file(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    (ctx / ".dockerignore").write_text("node_modules\nweird\\*name\n")
    (ctx / "node_modules").mkdir()
    (ctx / "node_modules" / "a.js").write_text("1")
    said = []
    before = images.build_hash(plan, {}, said.append)
    assert "every context file counts" in said[0]
    (ctx / "node_modules" / "a.js").write_text("22")
    assert images.build_hash(plan, {}, _quiet) != before


def test_an_ignore_file_past_the_work_budget_counts_every_file(fake_container, tmp_path,
                                                               monkeypatch):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    (ctx / ".dockerignore").write_text("node_modules\n*a*a*a*b\n")
    (ctx / "node_modules").mkdir()
    (ctx / "node_modules" / "a.js").write_text("1")
    for i in range(20):
        (ctx / f"file{i}.txt").write_text("x")
    monkeypatch.setattr(ignore, "WORK_MAX", 50)
    said = []
    before = images.build_hash(plan, {}, said.append)
    assert [line for line in said if "takes too long" in line
            and "every context file counts" in line]
    (ctx / "node_modules" / "a.js").write_text("22")        # now counted
    assert images.build_hash(plan, {}, _quiet) != before


def test_packages_with_build_need_the_own_base(tmp_path):
    with pytest.raises(images.ImageError, match="FROM gmlx.invalid/launch-pi:base"):
        _user_build(tmp_path, text="FROM debian\n", packages=["make"])
    _, plan = _user_build(tmp_path, packages=["make"])
    assert plan.packages == ["make"]


def test_base_packages_come_from_the_base_client(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM gmlx.invalid/launch-omp:base\n")
    cfg = LaunchContainerCfg(clients={"pi": LaunchClientCfg(build=str(ctx)),
                                      "omp": LaunchClientCfg(packages=["jq"])})
    plan = images.resolve_image("pi", cfg.for_client("pi"), cfg)
    assert plan.base_packages == {"omp": ["jq"]}


def test_build_path_rules(tmp_path):
    cfg = LaunchContainerCfg()
    with pytest.raises(images.ImageError, match="absolute path"):
        images.resolve_image("pi", LaunchClientCfg(build="ctx"), cfg)
    with pytest.raises(images.ImageError, match="does not exist"):
        images.resolve_image("pi", LaunchClientCfg(build=str(tmp_path / "no")), cfg)
    with pytest.raises(images.ImageError, match="no Containerfile"):
        images.resolve_image("pi", LaunchClientCfg(build=str(tmp_path)), cfg)
    big = tmp_path / "Big.containerfile"
    big.write_text("#" * images.CONTAINERFILE_MAX)
    with pytest.raises(images.ImageError, match="which Apple container cannot build"):
        images.resolve_image("pi", LaunchClientCfg(build=str(big)), cfg)


@pytest.mark.parametrize("layout", ["context in share", "file in share", "share in context",
                                    "file linked into share", "context in share, file linked out"])
def test_a_build_folder_the_client_can_write_is_refused(tmp_path, layout):
    """A guest edit would run at the next build, with the network and the
    builder, and later launches from other projects would build it too."""
    proj = tmp_path / "proj"
    ctx = proj / ".gmlx"
    ctx.mkdir(parents=True)
    (ctx / "Containerfile").write_text("FROM debian\n")
    build, share = str(ctx), str(proj)
    if layout == "file in share":
        build = str(ctx / "Containerfile")
    elif layout == "share in context":
        (proj / "Containerfile").write_text("FROM debian\n")
        build, share = str(proj), str(ctx)
    elif layout == "file linked into share":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "Containerfile").symlink_to(ctx / "Containerfile")
        build = str(outside)
    elif layout == "context in share, file linked out":
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (ctx / "Containerfile").rename(elsewhere / "Containerfile")
        (ctx / "Containerfile").symlink_to(elsewhere / "Containerfile")
    cfg = LaunchContainerCfg()
    with pytest.raises(images.ImageError, match="through the read-write share"):
        images.resolve_image("pi", LaunchClientCfg(build=build), cfg,
                             writable=[os.path.realpath(share)])
    other = tmp_path / "other"
    other.mkdir()
    plan = images.resolve_image("pi", LaunchClientCfg(build=build), cfg,
                                writable=[os.path.realpath(other)])
    assert plan.kind == "build"


def test_a_build_folder_in_the_launch_data_folder_is_refused():
    """Another client's private home, or a context that holds them all."""
    from gmlx.container import settings
    img = settings.private_home("pi") / "img"
    img.mkdir()
    (img / "Containerfile").write_text("FROM debian\n")
    above = Path(os.environ["XDG_DATA_HOME"])          # holds gmlx/launch
    (above / "Containerfile").write_text("FROM debian\n")
    cfg = LaunchContainerCfg()
    for build in (img, img / "Containerfile", above):
        with pytest.raises(images.ImageError, match="where the clients' private homes are"):
            images.resolve_image("claude-code", LaunchClientCfg(build=str(build)), cfg)


def test_a_build_path_through_a_link_the_client_can_change_is_refused(tmp_path):
    """The build reads the path as written, so a client that points a link
    on the way at a folder of its own changes what the next build runs."""
    from types import SimpleNamespace

    from gmlx.container import settings
    proj, ctx = tmp_path / "proj", tmp_path / "containers" / "box"
    ctx.mkdir(parents=True)
    proj.mkdir()
    (ctx / "Containerfile").write_text("FROM debian\n")
    (proj / "box").symlink_to(ctx)
    (tmp_path / "builds").symlink_to(proj)
    cfg = LaunchContainerCfg()
    for build in (proj / "box", proj / "box" / "Containerfile", tmp_path / "builds" / "box"):
        with pytest.raises(images.ImageError, match=r"build: path leads through .*/proj/box "
                                                    r"in the read-write share"):
            images.resolve_image("pi", LaunchClientCfg(build=str(build)), cfg,
                                 writable=[os.path.realpath(proj)])
    assert images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), cfg,
                                writable=[os.path.realpath(proj)]).kind == "build"
    real = os.path.realpath(proj)
    settings.record_shares(SimpleNamespace(mounts=[settings.Mount(real, real)]))
    with pytest.raises(images.ImageError, match=r"an earlier launch shared .* read-write, and "
                                                r"the pi build: path leads through") as e:
        images.resolve_image("pi", LaunchClientCfg(build=str(proj / "box")), cfg)
    assert str(e.value).endswith(settings.forget_step(real))
    assert images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), cfg).kind == "build"


def test_a_build_folder_an_earlier_launch_shared_is_refused(tmp_path):
    from types import SimpleNamespace

    from gmlx.container import settings
    proj = tmp_path / "proj"
    ctx = proj / ".gmlx"
    ctx.mkdir(parents=True)
    (ctx / "Containerfile").write_text("FROM debian\n")
    cfg = LaunchContainerCfg()
    assert images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), cfg).kind == "build"
    real = os.path.realpath(proj)
    settings.record_shares(SimpleNamespace(mounts=[settings.Mount(real, real)]))
    with pytest.raises(images.ImageError, match="an earlier launch shared .* read-write") as e:
        images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), cfg)
    assert str(e.value).endswith(settings.forget_step(real))
    # A read-only share leaves no record.
    other = tmp_path / "other"
    other.mkdir()
    (other / "Containerfile").write_text("FROM debian\n")
    settings.record_shares(SimpleNamespace(mounts=[
        settings.Mount(os.path.realpath(other), "/o", readonly=True)]))
    assert images.resolve_image("pi", LaunchClientCfg(build=str(other)), cfg).kind == "build"


@pytest.mark.parametrize("alias", ["build", "share"])
def test_a_firmlink_alias_never_hides_a_build_folder_in_a_share(tmp_path, alias):
    """realpath keeps /System/Volumes/Data, so the refusal compares the form
    macOS gives each path."""
    proj = tmp_path / "proj"
    ctx = proj / "ctx"
    ctx.mkdir(parents=True)
    (ctx / "Containerfile").write_text("FROM debian\n")
    data = "/System/Volumes/Data" + os.path.realpath(ctx)
    if not os.path.isdir(data):
        pytest.skip("no /System/Volumes/Data firmlink here")
    build, share = str(ctx), os.path.realpath(proj)
    if alias == "build":
        build = data
    else:
        share = "/System/Volumes/Data" + share
    with pytest.raises(images.ImageError, match="through the read-write share"):
        images.resolve_image("pi", LaunchClientCfg(build=build), LaunchContainerCfg(),
                             writable=[share])


def test_a_containerfile_that_is_a_pipe_is_refused_at_once(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    os.mkfifo(ctx / "Containerfile")
    with pytest.raises(images.ImageError, match="not a regular file"):
        images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), LaunchContainerCfg())


def test_a_rebuild_names_what_changed(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    said = []
    images.ensure_image(plan, say=said.append)
    assert not [line for line in said if "rebuilding because" in line]   # first build
    (ctx / "Containerfile").write_text("FROM debian\nRUN true\n")
    said.clear()
    images.ensure_image(plan, say=said.append)
    assert [line for line in said if "rebuilding because" in line] == [
        f"[launch] rebuilding because {ctx / 'Containerfile'} changed"]
    for name in "abcde":
        (ctx / name).write_text(name)
    said.clear()
    images.ensure_image(plan, say=said.append)
    (line,) = [line for line in said if "rebuilding because" in line]
    assert line == f"[launch] rebuilding because {ctx / 'a'}, {ctx / 'b'}, {ctx / 'c'} and 2 more changed"
    said.clear()
    images.ensure_image(plan, say=said.append)          # nothing changed, nothing built
    assert not [line for line in said if "rebuilding" in line]


def test_the_rebuild_reason_names_a_changed_base_and_escapes_names():
    old = {"containerfile": "/c/Containerfile", "context": "/c", "containerfile_sha256": "x",
           "bases": {"pi": D1}, "files": {"a": "1"}}
    new = dict(old, bases={"pi": D2}, files={"a": "1", "b\x1b[2Jc": "2"})
    assert images.rebuild_reason(old, new) == (
        "[launch] rebuilding because /c/b\\x1b[2Jc and the pi base image changed")
    assert images.rebuild_reason(None, new) is None
    assert images.rebuild_reason(old, old) is None


def test_records_are_private_and_never_written_through_a_link(fake_container, tmp_path):
    images.images_dir().mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    images._records_path().symlink_to(outside)
    old = os.umask(0o002)
    try:
        images._update_records(lambda records: records.update({"x": {"clients": ["pi"]}}))
        images._write_date(images._owed_path(), "2026-09-28T09:00:00Z")
    finally:
        os.umask(old)
    assert outside.read_text() == "keep"
    for path in (images._records_path(), images._owed_path()):
        assert not path.is_symlink()
        assert (os.stat(path).st_mode & 0o777) == 0o600, path


def test_shipped_containerfile_stays_under_the_limit_and_covers_every_client():
    text = images.SHIPPED_CONTAINERFILE.read_text()
    assert len(text.encode()) < images.CONTAINERFILE_MAX
    from gmlx.config import LAUNCH_CLIENTS
    preamble, named, last = images._stages(text)
    # A CLIENT without a stage would name an image on Docker Hub.
    shared = {"common", "python", "python-3.12"}
    assert set(named) == {*LAUNCH_CLIENTS, *images.RUNTIME_STAGES.values(), *shared}
    assert preamble[0] == "ARG CLIENT=common" and last[0] == "FROM ${CLIENT}"
    assert all(line.startswith("ARG ") for line in preamble)
    for client in [*LAUNCH_CLIENTS, *images.RUNTIME_STAGES.values()]:
        assert named[client][0] in shared, client
        assert images.shipped_version(client), client
    # An agent's key starts with agent-, which keeps its folders and images
    # apart from every stage's.
    assert not any(name.startswith("agent-") for name in named)
    assert set(images.CLIENT_BINARY) == set(LAUNCH_CLIENTS)
    assert set(images.RUNTIME_BINARY) == set(images.RUNTIME_STAGES.values())


def test_image_override_notes_unused_packages():
    plan = images.resolve_image("pi", LaunchClientCfg(packages=["make"]), LaunchContainerCfg(),
                                image_override="debian:12")
    assert plan.kind == "image" and "packages list is not used" in plan.notices[0]


# image: references

def test_missing_image_is_pulled_and_pinned(fake_container):
    fake_container.update(registry={"debian:12": {"digest": D1}})
    ready = images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"), say=_quiet)
    assert ["image", "pull", "debian:12"] in fake_container.log
    assert ready.action == "pulled"
    assert ready.run_ref == f"docker.io/library/debian@{D1}"
    assert ready.run_ref in fake_container.load()["images"]
    assert ready.tag == "debian:12"


@pytest.mark.parametrize("ref,server", [("ghcr.io/example/agent:1", "ghcr.io"),
                                        ("me/agent:1", "docker.io")])
def test_a_failed_pull_names_the_registry_login(fake_container, ref, server):
    with pytest.raises(images.ImageError) as e:
        images.ensure_image(images.ImagePlan("image", "pi", ref=ref), say=_quiet)
    assert str(e.value).endswith("Check the image reference. When the image is private, "
                                 "sign in to its registry with: container registry login "
                                 f"{server}")


def test_rebuild_pulls_again_and_drops_the_old_digest(fake_container):
    fake_container.update(registry={"debian:12": {"digest": D1}})
    first = images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"), say=_quiet)
    fake_container.update(registry={"debian:12": {"digest": D2}})
    second = images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"),
                                 rebuild=True, say=_quiet)
    store = fake_container.load()["images"]
    assert second.run_ref != first.run_ref
    assert first.run_ref not in store and "docker.io/library/debian:12" in store


def test_a_reference_that_moves_before_its_pin_is_refused(fake_container, monkeypatch):
    """A pull outside launch between the inspect and the tag would leave a
    digest reference that holds another image."""
    fake_container.update(registry={"me/box:1": _img(D1)})
    real_tag = cli.tag

    def moved_tag(source, target):
        state = fake_container.load()
        state["images"]["docker.io/me/box:1"] = _img(D2)
        fake_container.save(state)
        real_tag(source, target)
    monkeypatch.setattr(cli, "tag", moved_tag)
    with pytest.raises(images.ImageError, match=r"^me/box:1 changed while launch added its "
                                                r"digest reference, so launch did not use it\. "
                                                r"Launch again\.$"):
        images.ensure_image(images.ImagePlan("image", "pi", ref="me/box:1"), say=_quiet)
    assert f"docker.io/me/box@{D1}" not in fake_container.load()["images"]
    assert not _records()


def test_the_lookup_and_pull_of_an_image_wait_for_its_lock(fake_container):
    """Another launch's pull or cleanup of the repository finishes before
    this launch looks up the reference."""
    fake_container.update(registry={"me/box:1": _img(D1)})
    other = images.repo_lock("docker.io/me/box")
    waiting = threading.Event()
    result: list = []

    def say(line):
        if "waiting for another launch to finish with docker.io/me/box" in line:
            waiting.set()

    def run():
        result.append(images.ensure_image(images.ImagePlan("image", "pi", ref="me/box:1"),
                                          say=say))
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert waiting.wait(10)
        before = fake_container.calls("image")
    finally:
        other.release()
        worker.join(10)
    assert before == []
    assert result and result[0].run_ref == f"docker.io/me/box@{D1}"


@pytest.mark.parametrize("rebuild", [False, True])
def test_a_reference_you_pulled_stays_yours_under_rebuild(fake_container, rebuild):
    fake_container.update(registry={"node:22": _img(D2)}, images={"node:22": _img(D1)})
    images.forget_unnamed(_config(pi={"image": "node:22"}), _quiet)
    images.ensure_image(images.ImagePlan("image", "pi", ref="node:22"), rebuild=rebuild,
                        say=_quiet)
    images.forget_unnamed(_config(), _quiet)
    assert "docker.io/library/node:22" in fake_container.load()["images"]


def test_a_reference_launch_pulled_goes_under_rebuild_too(fake_container):
    fake_container.update(registry={"node:22": _img(D1)})
    images.forget_unnamed(_config(pi={"image": "node:22"}), _quiet)
    images.ensure_image(images.ImagePlan("image", "pi", ref="node:22"), rebuild=True,
                        say=_quiet)
    images.forget_unnamed(_config(), _quiet)
    assert "docker.io/library/node:22" not in fake_container.load()["images"]


@pytest.mark.parametrize("noted", [True, False])
def test_a_digest_reference_that_holds_another_image_is_tagged_again(fake_container, noted):
    # A manual `container image tag` can put the digest reference on
    # another image. Launch runs the image of the digest the name gives.
    ref, run_ref = "docker.io/me/box:1", "docker.io/me/box@" + D1
    fake_container.update(images={ref: _img(D1), run_ref: _img(D2)})
    if noted:
        images._write_pin(ref, run_ref, fetched=True)
    ready = images.ensure_image(images.ImagePlan("image", "pi", ref=ref), say=_quiet)
    assert (ready.info.digest, ready.run_ref) == (D1, run_ref)
    assert fake_container.load()["images"][run_ref]["digest"] == D1
    assert ["image", "tag", ref, run_ref] in fake_container.calls("image")


def test_non_arm64_image_is_refused(fake_container):
    fake_container.update(images={"x86:1": _img(arch=["linux/amd64"])})
    with pytest.raises(images.ImageError, match=r"no linux/arm64 variant \(linux/amd64\)"):
        images.ensure_image(images.ImagePlan("image", "pi", ref="x86:1"), say=_quiet)


# The one-time command check

def _ready(fake_container, kind="image", client="claude-code"):
    fake_container.update(images={"x:1": _img()})
    return images.ReadyImage(kind, "x:1", cli.image_info("x:1"), "x@" + D1, "found", client)


def test_check_runs_by_digest_and_is_cached(fake_container):
    ready = _ready(fake_container)
    images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    runs = fake_container.calls("run")
    assert len(runs) == 1
    assert runs[0][-3:] == ["x@" + D1, "--check", "claude"]
    assert "--network" in runs[0] and "type=bind,source=/rt,target=/opt/gmlx,readonly" in runs[0]
    images.check_command(ready, "my-client", "/rt", shell=False, say=_quiet)
    assert len(fake_container.calls("run")) == 2       # a new command checks again


def test_the_check_runs_as_root_as_the_session_does(fake_container):
    # An image with a USER would check a root-only command as that user.
    images.check_command(_ready(fake_container), "claude", "/rt", shell=False, say=_quiet)
    (run,) = fake_container.calls("run")
    assert run[run.index("--uid"):run.index("--uid") + 4] == ["--uid", "0", "--gid", "0"]


def test_missing_command_refuses_or_warns_under_shell(fake_container):
    ready = _ready(fake_container)
    msg = "claude is not on the image's PATH (/usr/bin)."
    fake_container.update(checks={"claude": [127, f"[launch] {msg}"]})
    with pytest.raises(images.ImageError, match="not on the image's PATH"):
        images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    said = []
    images.check_command(ready, "claude", "/rt", shell=True, say=said.append)
    assert said == [f"[launch] warning: {msg}"]


def test_the_missing_command_hint_names_the_targets_config_key(fake_container):
    """The guest entry knows no config key and prints a placeholder, which
    the Mac fills in for a client, an agent and a runtime agent."""
    import re
    tail = "Install it in the image, or set launch.container.clients.<client>.command."
    fake_container.update(checks={
        "uv": [127, f"[launch] uv is not on the image's PATH (/usr/bin). {tail}"],
        "bot": [127, f"[launch] bot is not on the image's PATH (/usr/bin). {tail}"]})
    with pytest.raises(images.ImageError, match=re.escape(
            "PATH (/usr/bin). Install it in the image, or set launch.container.clients.pi.command.")):
        images.check_command(_ready(fake_container, client="pi"), "bot", "/rt", shell=False,
                             say=_quiet)
    with pytest.raises(images.ImageError, match=re.escape("or set launch.agents.bot.command.")):
        images.check_command(_ready(fake_container, client="agent-bot"), "bot", "/rt",
                             shell=False, say=_quiet)
    with pytest.raises(images.ImageError, match=re.escape(
            "Install it in the image, or remove launch.agents.bot.runtime, so the command runs "
            "as written, without uv.")):
        images.check_command(_ready(fake_container, client="agent-bot"), "uv", "/rt",
                             shell=False, say=_quiet, runtime=True)
    said = []
    images.check_command(_ready(fake_container, client="agent-bot"), "uv", "/rt", shell=True,
                         say=said.append, runtime=True)
    assert said[0].endswith("or remove launch.agents.bot.runtime, so the command runs as "
                            "written, without uv.")


def test_shipped_images_skip_the_check_only_for_their_own_client(fake_container):
    ready = _ready(fake_container, "shipped")
    images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    assert fake_container.calls("run") == []
    # The shipped runtime image has the uv and sh that a runtime agent needs.
    runtime = _ready(fake_container, "shipped", client="runtime-python")
    for word in ("uv", "sh"):
        images.check_command(runtime, word, "/rt", shell=False, say=_quiet, runtime=True)
    assert fake_container.calls("run") == []
    # A command: list on the shipped image names a command the image may lack.
    fake_container.update(checks={"start.sh": [127, "[launch] start.sh is not there."]})
    with pytest.raises(images.ImageError, match="start.sh is not there"):
        images.check_command(ready, "start.sh", "/rt", shell=False, say=_quiet)


def test_a_command_without_the_execute_bit_refuses_like_a_missing_one(fake_container):
    ready = _ready(fake_container)
    msg = "/usr/local/bin/start.sh has no execute bit. Run chmod 755 on it."
    fake_container.update(checks={"start.sh": [126, f"[launch] {msg}"]})
    with pytest.raises(images.ImageError, match="has no execute bit"):
        images.check_command(ready, "start.sh", "/rt", shell=False, say=_quiet)
    said = []
    images.check_command(ready, "start.sh", "/rt", shell=True, say=said.append)
    assert said == [f"[launch] warning: {msg}"]
    # A failed check is never remembered, so the fixed image passes next time.
    fake_container.update(checks={})
    images.check_command(ready, "start.sh", "/rt", shell=False, say=_quiet)
    images.check_command(ready, "start.sh", "/rt", shell=False, say=_quiet)
    assert len(fake_container.calls("run")) == 3


def test_other_check_failures_refuse_even_under_shell(fake_container):
    ready = _ready(fake_container)
    fake_container.update(checks={"x": [125, "[launch] cannot listen."]})
    with pytest.raises(images.ImageError, match="failed \\(exit 125\\)"):
        images.check_command(ready, "x", "/rt", shell=True, say=_quiet)


def test_the_check_container_has_a_name_and_is_removed_on_a_timeout(fake_container,
                                                                   monkeypatch):
    ready = _ready(fake_container)
    images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    (run,) = fake_container.calls("run")
    name = run[run.index("--name") + 1]
    assert name.startswith("gmlx-check-")
    monkeypatch.setattr(cli, "CHECK_TIMEOUT", 0.001)
    with pytest.raises(cli.Unavailable, match="gave no answer"):
        images.check_command(ready, "other", "/rt", shell=False, say=_quiet)
    (delete,) = fake_container.calls("delete")
    assert delete[:2] == ["delete", "--force"] and delete[2].startswith("gmlx-check-")


def test_a_query_that_times_out_names_the_restart_apart_from_its_reason(fake_container,
                                                                      monkeypatch):
    """A message that gives a step of its own can use the reason alone."""
    def hang(*_a, **_k):
        raise subprocess.TimeoutExpired("container", 60)
    monkeypatch.setattr(cli.subprocess, "run", hang)
    with pytest.raises(cli.Unavailable) as e:
        cli.containers()
    assert e.value.reason == ("`container ls --all --format` gave no answer in 60 s, so "
                              "the container service may be stuck")
    assert str(e.value) == (f"{e.value.reason}. Restart it with: container system stop "
                            "&& container system start")


def test_a_builder_stop_that_meets_a_stuck_service_names_one_step(fake_container,
                                                                 no_other_builds,
                                                                 monkeypatch):
    """The stop of the builder waits for a service that gives no answer. A
    restart of the service stops the builder too, so the warning gives that
    step only."""
    real_run = cli.subprocess.run

    def run(argv, *a, **k):
        if argv[1:] == ["builder", "stop"]:
            raise subprocess.TimeoutExpired(argv, 60)
        return real_run(argv, *a, **k)
    monkeypatch.setattr(cli.subprocess, "run", run)
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert [line for line in said if "image builder" in line] == [
        "[launch] warning: could not stop the image builder, because `container builder "
        "stop` gave no answer in 60 s, so the container service may be stuck. Restart it "
        "with: container system stop && container system start"]


def test_image_command_forms(fake_container):
    fake_container.update(images={"x:1": _img(entrypoint=["bash", "start.sh"], cmd=["--x"],
                                              workdir="/app/backend")})
    ready = images.ReadyImage("image", "x:1", cli.image_info("x:1"), "x@" + D1, "found")
    assert images.image_command(ready, None, ["claude"], ["--continue"]) == (
        ["claude", "--continue"], None)
    assert images.image_command(ready, ["my", "-v"], ["claude"], ["a"]) == (["my", "-v", "a"], None)
    assert images.image_command(ready, "image", ["claude"], []) == (
        ["bash", "start.sh", "--x"], "/app/backend")
    assert images.image_command(ready, "image", ["claude"], ["--y"]) == (
        ["bash", "start.sh", "--y"], "/app/backend")


def test_an_empty_entrypoint_runs_the_image_cmd(fake_container):
    fake_container.update(images={"x:1": _img(entrypoint=[""], cmd=["myapp", "--serve"])})
    info = cli.image_info("x:1")
    assert info.entrypoint is None
    ready = images.ReadyImage("image", "x:1", info, "x@" + D1, "found")
    assert images.image_command(ready, "image", ["claude"], []) == (["myapp", "--serve"], None)


def test_describe_and_age_note(fake_container):
    fake_container.update(images={"x:1": _img(created="2020-01-01T00:00:00Z")})
    built = datetime(2026, 8, 1, tzinfo=timezone.utc)
    ready = images.ReadyImage("shipped", "x:1", cli.image_info("x:1"), "x@" + D1, "found",
                              fetched=built)
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    assert images.describe(ready, now) == "[launch] image x:1, built 57 days ago"
    note = images.image_age_note(ready, now)
    assert note == ("[launch] launch built this image 57 days ago. --rebuild builds it again "
                    "with current packages, which ends this note.")
    assert (note.key, note.every) == ("age:x:1", 24 * 3600)
    assert images.image_age_note(ready, datetime(2026, 8, 20, tzinfo=timezone.utc)) is None
    pulled = images.ReadyImage("image", "x:1", ready.info, "x@" + D1, "found", fetched=built)
    assert images.describe(pulled, now) == "[launch] image x:1, pulled 57 days ago"
    pi = images.ReadyImage("shipped", "x:1", ready.info, "x@" + D1, "found", "pi", built)
    version = images.shipped_version("pi")
    assert images.describe(pi, now) == f"[launch] image x:1 with pi {version}, built 57 days ago"
    assert "--rebuild pulls it again, which ends this note." in images.image_age_note(pulled, now)


def test_the_age_counts_from_the_last_pull_not_the_creation_date(fake_container, monkeypatch):
    """A registry image keeps its creation date, so --rebuild could never
    end a note that counted from it."""
    fake_container.update(registry={"box:1": _img(created="2026-01-01T00:00:00Z")})
    plan = images.ImagePlan("image", "pi", ref="box:1")
    clock = [datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()]
    monkeypatch.setattr(images.time, "time", lambda: clock[0])
    first = images.ensure_image(plan, say=_quiet)
    assert first.fetched == datetime(2026, 9, 1, tzinfo=timezone.utc)
    clock[0] += 40 * 86400
    later = images.ensure_image(plan, say=_quiet)
    now = datetime.fromtimestamp(clock[0], timezone.utc)
    assert later.fetched == first.fetched and "40 days" in images.image_age_note(later, now)
    again = images.ensure_image(plan, rebuild=True, say=_quiet)       # the same digest
    assert again.fetched == now and images.image_age_note(again, now) is None


def test_an_image_found_without_a_note_counts_from_now(fake_container):
    fake_container.update(images={"box:1": _img(created="2026-01-01T00:00:00Z")})
    ready = images.ensure_image(images.ImagePlan("image", "pi", ref="box:1"), say=_quiet)
    assert images._age_days(ready, None) == 0


# Images the config no longer names

def _config(_agents=None, **clients):
    return LaunchCfg(container=LaunchContainerCfg(
        clients={c: LaunchClientCfg(**kw) for c, kw in clients.items()}), agents=_agents or {})


def test_normalized_names_a_reference_as_the_store_does():
    assert images.normalized("debian:12") == "docker.io/library/debian:12"
    assert images.normalized("me/box") == "docker.io/me/box:latest"
    assert images.normalized("ghcr.io/me/box:1") == "ghcr.io/me/box:1"
    assert images.normalized("localhost:5000/box") == "localhost:5000/box:latest"
    assert images.normalized("me/box@" + D1) == "docker.io/me/box@" + D1


def test_a_build_repository_goes_once_build_is_removed(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    with_build = _config(pi={"build": str(ctx)})
    images.forget_unnamed(with_build, _quiet)               # the first launch records
    ready = images.ensure_image(plan, say=_quiet)
    images.forget_unnamed(with_build, _quiet)
    assert ready.tag in fake_container.load()["images"]
    images.forget_unnamed(_config(), _quiet)
    store = fake_container.load()["images"]
    assert not any(name.startswith(images.build_repo("pi")) for name in store), store


def test_a_pulled_reference_goes_once_no_client_names_it(fake_container):
    fake_container.update(registry={"me/box:1": _img(D1), "me/box:2": _img(D2)},
                          images={"me/mine:1": _img("sha256:" + "5" * 64)})
    images.forget_unnamed(_config(opencode={"image": "me/box:1"}), _quiet)
    one = images.ensure_image(images.ImagePlan("image", "opencode", ref="me/box:1"), say=_quiet)
    images.ensure_image(images.ImagePlan("image", "opencode", ref="me/box:2"), say=_quiet)
    images.forget_unnamed(_config(opencode={"image": "me/box:2"}), _quiet)
    store = fake_container.load()["images"]
    assert "docker.io/me/box:1" not in store and one.run_ref not in store
    assert "docker.io/me/box:2" in store and "docker.io/me/mine:1" in store


def test_a_reference_you_pulled_or_a_running_container_uses_stays(fake_container):
    fake_container.update(registry={"me/box:2": _img(D2)},
                          images={"me/box:1": _img(D1), "me/run:1": _img("sha256:" + "6" * 64)})
    images.forget_unnamed(_config(opencode={"image": "me/box:1"}, pi={"image": "me/run:1"}),
                          _quiet)
    images.ensure_image(images.ImagePlan("image", "opencode", ref="me/box:1"), say=_quiet)
    fake_container.update(registry={"me/run:1": _img("sha256:" + "6" * 64)})
    state = fake_container.load()
    del state["images"]["docker.io/me/run:1"]
    fake_container.save(state)
    images.ensure_image(images.ImagePlan("image", "pi", ref="me/run:1"), say=_quiet)
    fake_container.update(containers=[{"name": "c", "image": "docker.io/me/run:1"}])
    images.forget_unnamed(_config(), _quiet)
    store = fake_container.load()["images"]
    assert "docker.io/me/box:1" in store                    # found, never pulled by launch
    assert "docker.io/me/run:1" in store                    # a running container uses it


def test_disk_report_counts_pulled_images_and_names_the_unused(fake_container):
    cfg = _config(opencode={"image": "me/box:2"})
    shipped = images.shipped_tag("pi", [])
    d3, d4 = "sha256:" + "3" * 64, "sha256:" + "4" * 64
    fake_container.update(
        images={shipped: _img(D1, size=3 << 30), images.base_ref("pi"): _img(D1, size=3 << 30),
                "gmlx.invalid/launch-pi:0ld": _img(d3, size=1 << 30),
                "gmlx.invalid/launch-pi-build:x": _img(d4, size=1 << 30),
                "debian:12": _img("sha256:" + "7" * 64, size=1 << 30)},
        registry={"me/box:1": _img("sha256:" + "8" * 64, size=1 << 20),
                  "me/box:2": _img(D2, size=2 << 20)})
    for ref in ("me/box:1", "me/box:2"):
        images.ensure_image(images.ImagePlan("image", "opencode", ref=ref), say=_quiet)
    count, size, unused = images.disk_report(cfg)
    assert count == 5 and size == (5 << 30) + (3 << 20)    # debian:12 is not launch's
    assert unused == ["docker.io/me/box:1", "gmlx.invalid/launch-pi-build:x",
                      "gmlx.invalid/launch-pi:0ld"]
    assert images.disk_report(None)[2] == []


# Locks

def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - reported to the test
            out["error"] = e
    t = threading.Thread(target=run)
    t.start()
    return t, out


class _Said(list):
    """The lines a call says, which another thread can wait for."""

    def __init__(self):
        super().__init__()
        self._changed = threading.Condition()

    def append(self, line) -> None:
        with self._changed:
            super().append(line)
            self._changed.notify_all()

    def wait_for(self, text: str) -> None:
        with self._changed:
            # The timeout only ends a test whose call never says the line.
            assert self._changed.wait_for(lambda: any(text in line for line in self), 10), \
                f"no {text!r} in {list(self)}"


def test_second_launch_waits_for_the_build_and_builds_nothing(fake_container):
    tag = images.shipped_tag("pi", [])
    held = images.repo_lock("gmlx.invalid/launch-pi")
    said = _Said()
    t, out = _in_thread(lambda: images.ensure_image(images.ImagePlan("shipped", "pi"),
                                                    say=said.append))
    said.wait_for("waiting for another launch")
    fake_container.update(images={tag: _img(D1), "gmlx.invalid/launch-pi:base": _img(D1)})
    held.release()
    t.join(10)
    assert out["value"].action == "found"
    assert "builds" not in fake_container.load()


def test_base_rebuild_waits_while_a_user_build_holds_it_shared(fake_container):
    shared = images.repo_lock("gmlx.invalid/launch-pi", shared=True)
    said = _Said()
    t, out = _in_thread(lambda: images.ensure_image(images.ImagePlan("shipped", "pi"),
                                                    rebuild=True, say=said.append))
    said.wait_for("waiting for another launch")
    assert "builds" not in fake_container.load()
    shared.release()
    t.join(10)
    assert out["value"].action == "built"


def test_user_build_rechecks_a_base_that_moved(fake_container, tmp_path, monkeypatch):
    _, plan = _user_build(tmp_path)
    real_lock = images.repo_lock
    moved = {"done": False}

    def lock(repo, *, shared=False, say=None):
        taken = real_lock(repo, shared=shared, say=say)
        if shared and not moved["done"]:
            moved["done"] = True                  # another launch moves :base
            state = fake_container.load()
            state["images"]["gmlx.invalid/launch-pi:base"] = _img(D2)
            fake_container.save(state)
        return taken
    monkeypatch.setattr(images, "repo_lock", lock)
    images.ensure_image(plan, say=_quiet)
    tag = images.shipped_tag("pi", [])
    assert fake_container.log.count(["image", "tag", tag, "gmlx.invalid/launch-pi:base"]) == 1
    assert len(fake_container.load()["builds"]) == 2


def test_lock_file_descriptor_is_not_inherited(tmp_path):
    lock = FileLock(tmp_path / "x.lock")
    assert not os.get_inheritable(lock.fd)
    lock.release()


# Service errors, the reference records and the first-run lines

def test_image_info_raises_on_a_service_error(fake_container):
    fake_container.update(inspect_error="XPC connection error: the service is not running")
    with pytest.raises(cli.ContainerError, match="XPC connection error"):
        cli.image_info("x:1")
    # So a launch never pulls again because the service failed.
    fake_container.update(registry={"debian:12": {"digest": D1}})
    with pytest.raises(cli.ContainerError):
        images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"), say=_quiet)
    assert not fake_container.calls("image", "pull")


def test_an_os_error_becomes_a_container_error(fake_container, monkeypatch):
    def boom(*a, **k):
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(cli.subprocess, "run", boom)
    with pytest.raises(cli.ContainerError, match="Too many open files"):
        cli.containers()


def _records():
    return images._update_records(lambda r: None)


def test_a_digest_reference_you_added_is_never_recorded(fake_container):
    tag = images.shipped_tag("pi", [])
    run_ref = f"gmlx.invalid/launch-pi@{D1}"
    fake_container.update(images={tag: _img(D1), "gmlx.invalid/launch-pi:base": _img(D1),
                                  run_ref: _img(D1)})
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert run_ref not in _records()
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert run_ref in fake_container.load()["images"]


def test_a_shared_reference_goes_once_no_client_uses_it(fake_container):
    fake_container.update(registry={"debian:12": {"digest": D1}})
    first = images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"), say=_quiet)
    images.ensure_image(images.ImagePlan("image", "omp", ref="debian:12"), say=_quiet)
    assert set(_records()[first.run_ref]["clients"]) == {"pi", "omp"}
    _set_pin_pid(first.run_ref, 999999)          # both launches are gone
    fake_container.update(registry={"debian:12": {"digest": D2}})
    images.ensure_image(images.ImagePlan("image", "pi", ref="debian:12"), rebuild=True,
                        say=_quiet)
    assert first.run_ref in fake_container.load()["images"]       # omp still uses it
    assert _records()[first.run_ref]["clients"] == ["omp"]
    images.ensure_image(images.ImagePlan("image", "omp", ref="debian:12"), rebuild=True,
                        say=_quiet)
    assert first.run_ref not in fake_container.load()["images"]
    assert first.run_ref not in _records()


def test_an_image_found_through_its_pin_runs_no_pin_or_sweep(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert images._read_pins()[first.tag]["pin"] == first.run_ref
    fake_container.update(log=[])
    again = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert (again.action, again.run_ref) == ("found", first.run_ref)
    assert fake_container.log == [["image", "inspect", first.run_ref],
                                  ["image", "inspect", images.base_ref("pi")]]
    assert os.getpid() in _records()[first.run_ref]["pids"]


def test_a_pruned_pin_is_added_again(fake_container):
    """`container image prune` deletes every reference without a tag, and
    so every digest reference, while the tags stay."""
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    state = fake_container.load()
    del state["images"][first.run_ref]
    fake_container.save(state)
    again = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert again.run_ref == first.run_ref and again.action == "found"
    assert first.run_ref in fake_container.load()["images"]
    assert fake_container.calls("image", "tag")[-1] == ["image", "tag", first.tag,
                                                         first.run_ref]


def test_records_drop_dead_pids_and_gone_images(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    gone = "gmlx.invalid/launch-pi@" + D2

    def plant(records):
        records[gone] = {"clients": ["omp"], "pids": []}
        records[first.run_ref]["pids"] = [999999, os.getpid()]
    images._update_records(plant)
    # Without the note of its pin, the image is pinned and cleaned again.
    images._pins_path().unlink()
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    records = _records()
    assert gone not in records
    assert records[first.run_ref]["pids"] == [os.getpid()]


def test_a_cleanup_failure_only_warns(fake_container, monkeypatch):
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)

    def fail():
        raise cli.ContainerError("`container image list` gave no answer in 60 s.")
    monkeypatch.setattr(cli, "image_names", fail)
    said = []
    ready = images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]),
                                say=said.append)
    assert ready.action == "built"
    assert any("warning: could not delete older images" in line for line in said)


def test_a_refused_delete_does_not_fail_the_launch(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    fake_container.update(refuse_delete=[first.run_ref])
    ready = images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert ready.action == "built"
    assert _records()[first.run_ref]["clients"] == []
    fake_container.update(refuse_delete=[])
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=_quiet)
    assert first.run_ref not in fake_container.load()["images"]


def test_pending_work(fake_container, tmp_path):
    shipped = images.ImagePlan("shipped", "pi")
    assert images.pending_work(shipped, False) == "build"
    images.ensure_image(shipped, say=_quiet)
    assert images.pending_work(shipped, False) is None
    assert images.pending_work(shipped, True) == "build"
    pulled = images.ImagePlan("image", "pi", ref="debian:12")
    assert images.pending_work(pulled, False) == "pull"
    fake_container.update(registry={"debian:12": {"digest": D1}})
    images.ensure_image(pulled, say=_quiet)
    assert images.pending_work(pulled, False) is None
    assert images.pending_work(pulled, True) == "pull"
    _, plan = _user_build(tmp_path)
    assert images.pending_work(plan, False) == "build"
    images.ensure_image(plan, say=_quiet)
    assert images.pending_work(plan, False) is None
    builds = len(fake_container.load()["builds"])
    assert images.pending_work(plan, False) is None and len(fake_container.load()["builds"]) == builds


def test_the_step_goes_on_the_first_build_line_only(fake_container, tmp_path):
    _, plan = _user_build(tmp_path)
    said = []
    images.ensure_image(plan, say=said.append, step="step 2 of 3")
    builds = [line for line in said if "building" in line]
    assert len(builds) == 2
    assert builds[0] == ("[launch] step 2 of 3: building the pi image, which takes a few "
                         "minutes. Later launches reuse it.")
    assert builds[1].startswith("[launch] building ")
    assert sum("step 2 of 3" in line for line in said) == 1
    assert ("[launch] the build first downloads about 80 MB for the node:24-trixie-slim "
            "base image") in said


def test_the_node_download_is_named_only_when_it_happens(fake_container):
    fake_container.update(images={images._node_base(): _img(D2)})
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert "[launch] building the pi image, which takes a few minutes. Later launches reuse " \
           "it." in said
    assert not any("downloads" in line for line in said)


def test_the_shipped_image_installs_only_pinned_versions():
    """The base by digest, each client by version, each download by version
    and sha256, so a build installs what gmlx names."""
    import re
    text = images.SHIPPED_CONTAINERFILE.read_text()
    assert re.fullmatch(r"docker\.io/library/node:24-trixie-slim@sha256:[0-9a-f]{64}",
                        images._node_base())
    assert re.search(r"^ARG PIP_VERSION=\d+(\.\d+)+$", text, re.M)
    assert "releases/latest" not in text and "url_effective" not in text
    for version in re.findall(r"^ARG VERSION=(\S+)$", text, re.M):
        assert re.fullmatch(r"\d[\w.-]*", version), version
    for line in re.findall(r"npm install -g [^&;]*", text):
        for pkg in line.split()[3:]:
            assert re.search(r".@\$VERSION$", pkg), pkg
    installs = re.findall(r"pip install (?:--no-cache-dir|--python \S+) [^;\n]*", text)
    assert len(installs) == 6, installs
    for line in installs:
        pkgs = [w for w in line.split()[2:]
                if not w.startswith(("-", "http", "\\", "/"))]
        # A package with an extra, such as "hermes-agent[mcp]==$VERSION",
        # is quoted for the shell.
        pin = r"==([\w.]+|\$(PIP_)?VERSION)"
        assert pkgs and all(re.fullmatch(rf'[\w-]+{pin}|"[\w-]+\[[\w,-]+\]{pin}"', w)
                            for w in pkgs), line
    urls = re.findall(r'"(https://github\.com/[^"]+)"', text)
    assert len(urls) == 4 and all(re.search(r"/releases/download/v?\$VERSION/", u)
                                  for u in urls), urls
    assert len(re.findall(r'echo "[0-9a-f]{64}  \S+" \\\n\s*\| sha256sum -c -', text)) == 4


def test_the_shipped_image_upgrades_the_base_packages():
    """The digest fixes the base image, so only an upgrade brings the
    security updates of its own packages, such as libc6, to a rebuild."""
    _, named, _ = images._stages(images.SHIPPED_CONTAINERFILE.read_text())
    run = " ".join(named["common"][1])
    assert 0 <= run.find("apt-get update;") < run.find("apt-get upgrade -y;") < run.find(
        "apt-get install")


def test_the_shipped_layers_share_the_common_packages():
    _, named, last = images._stages(images.SHIPPED_CONTAINERFILE.read_text())
    assert named["common"][0] == images._node_base() and named["python"][0] == "common"
    assert {named[c][0] for c in ("hermes", "runtime-python")} == {"python"}
    # Elia and Open WebUI need a Python older than Debian's, and the uv of
    # the runtime stage installs it.
    assert named["python-3.12"][0] == "runtime-python"
    assert {named[c][0] for c in ("elia", "open-webui")} == {"python-3.12"}
    assert not any("EXTRA_PACKAGES" in line for _, lines in named.values() for line in lines)
    assert any("$EXTRA_PACKAGES" in line for line in last)


def test_the_install_table_matches_the_shipped_recipe():
    """docs/container-images.md gives each client's install line for a
    Containerfile of your own."""
    import re
    table = (Path(__file__).parents[2] / "docs" / "container-images.md").read_text()
    rows = dict(re.findall(r"^\| `([a-z-]+)` \| `([^`]+)` \|$", table, re.M))
    _, named, _ = images._stages(images.SHIPPED_CONTAINERFILE.read_text())
    assert len(rows) == 7
    for client, line in rows.items():
        stage = "\n".join(named[client][1])
        assert line.replace("<version>", "$VERSION") in stage, client


_SHIPPED = images.SHIPPED_CONTAINERFILE


def _recipe(tmp_path, monkeypatch, *edits):
    """Point the shipped Containerfile at a copy with ``edits``, pairs of
    old and new text, applied."""
    text = _SHIPPED.read_text()
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "files" / "Containerfile"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    monkeypatch.setattr(images, "SHIPPED_CONTAINERFILE", path)


def test_one_clients_pin_moves_only_its_own_tag(tmp_path, monkeypatch):
    from gmlx.config import LAUNCH_CLIENTS
    before = {c: images.shipped_hash(c, []) for c in LAUNCH_CLIENTS}
    _recipe(tmp_path, monkeypatch, ("ARG VERSION=2.1.283", "ARG VERSION=2.1.290"),
            ("# The image `gmlx", "# A comment changes nothing.\n# The image `gmlx"))
    after = {c: images.shipped_hash(c, []) for c in LAUNCH_CLIENTS}
    assert [c for c in LAUNCH_CLIENTS if after[c] != before[c]] == ["claude-code"]
    _recipe(tmp_path, monkeypatch, ("python3 python3-venv", "python3 python3-venv make"))
    moved = {c for c in LAUNCH_CLIENTS if images.shipped_hash(c, []) != before[c]}
    assert moved == {"hermes", "elia", "open-webui"}
    _recipe(tmp_path, monkeypatch, ("ARG VERSION=0.12.22", "ARG VERSION=0.12.23"))
    assert {c for c in LAUNCH_CLIENTS
            if images.shipped_hash(c, []) != before[c]} == {"elia", "open-webui"}
    _recipe(tmp_path, monkeypatch, ("less procps", "less procps jq"))
    assert all(images.shipped_hash(c, []) != before[c] for c in LAUNCH_CLIENTS)


def test_a_shipped_rebuild_says_why(fake_container, tmp_path, monkeypatch):
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert not any("rebuilding" in line for line in said)      # a first build
    said.clear()
    _recipe(tmp_path, monkeypatch, ("ARG VERSION=1.0.1", "ARG VERSION=1.0.2"))
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=said.append)
    assert said[0] == ("[launch] rebuilding because gmlx moved pi from 1.0.1 to 1.0.2 and the "
                       "packages list changed")
    said.clear()
    _recipe(tmp_path, monkeypatch, ("ARG VERSION=1.0.1", "ARG VERSION=1.0.2"),
            ("less procps", "less procps jq"))
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=said.append)
    assert said[0] == ("[launch] rebuilding because gmlx updated the layers that pi shares "
                       "with other clients")
    said.clear()
    state = fake_container.load()
    state["images"] = {}
    fake_container.save(state)
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=said.append)
    assert said[0] == ("[launch] rebuilding because the pi image is no longer in the image "
                       "store")
    said.clear()
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), rebuild=True,
                        say=said.append)
    assert not any("rebuilding" in line for line in said)      # asked for


def test_an_upgrade_from_a_build_without_a_recipe_record_says_why(fake_container):
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    images._manifest_path(images.recipe_repo("pi"), "recipe").unlink()
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make"]), say=said.append)
    assert said[0] == "[launch] rebuilding because gmlx updated the pi recipe"


def test_the_node_download_is_named_until_a_build_completes(fake_container, tmp_path,
                                                            monkeypatch):
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert any("downloads about 80 MB" in line for line in said)
    said.clear()
    images.ensure_image(images.ImagePlan("shipped", "omp"), say=said.append)
    assert any("building the omp image" in line for line in said)
    assert not any("downloads" in line for line in said)
    said.clear()
    digest = images._node_base().split("@")[1]
    _recipe(tmp_path, monkeypatch, (digest, "sha256:" + "0" * 64))
    images.ensure_image(images.ImagePlan("shipped", "omp"), say=said.append)
    assert any("downloads about 80 MB" in line for line in said)


def test_the_hash_walk_keeps_going_past_an_unreadable_folder(fake_container, tmp_path):
    ctx, plan = _user_build(tmp_path, text="FROM debian\n")
    locked = ctx / "locked"
    locked.mkdir()
    (locked / "a").write_text("1")
    locked.chmod(0o300)                              # listable, not searchable
    closed = ctx / "closed"
    closed.mkdir()
    closed.chmod(0)
    try:
        files = dict(images.context_files(ctx, None))
        assert files["locked"] is None or "locked/a" in files
        assert "closed" in files and files["closed"] is None
        images.build_hash(plan, {}, _quiet)
    finally:
        locked.chmod(0o755)
        closed.chmod(0o755)


def test_an_excluded_folder_is_pruned_when_no_exception_reaches_below(fake_container, tmp_path,
                                                                      monkeypatch):
    ctx, _plan = _user_build(tmp_path, text="FROM debian\n")
    (ctx / "node_modules" / "deep").mkdir(parents=True)
    (ctx / "node_modules" / "deep" / "x.js").write_text("1")
    (ctx / "keep.txt").write_text("1")
    seen = []
    real = images.os.walk

    def walk(top, **kw):
        for root, dirs, files in real(top, **kw):
            seen.append(root)
            yield root, dirs, files
    monkeypatch.setattr(images.os, "walk", walk)
    matcher = images.ignore.Matcher(["node_modules", "!keep.txt"])
    files = dict(images.context_files(ctx, matcher))
    assert "keep.txt" in files
    assert not any("node_modules" in root for root in seen)


def test_the_launch_tests_get_a_short_temporary_folder(fake_container, short_tmpdir):
    assert os.environ["TMPDIR"] == str(short_tmpdir)
    assert str(short_tmpdir).startswith("/tmp/gmlx-t-") and len(str(short_tmpdir)) < 30


def test_query_timeout_shortens_queries_in_its_block(fake_container, monkeypatch):
    seen = []
    real = cli.subprocess.run

    def run(argv, **kw):
        seen.append(kw.get("timeout"))
        return real(argv, **kw)
    monkeypatch.setattr(cli.subprocess, "run", run)
    with cli.query_timeout(5):
        cli.containers()
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])   # runs with no timeout
    cli.containers()
    assert seen == [5, cli.QUERY_TIMEOUT] and fake_container.calls("build")

    def slow(argv, **kw):
        raise cli.subprocess.TimeoutExpired(argv, kw["timeout"])
    monkeypatch.setattr(cli.subprocess, "run", slow)
    with cli.query_timeout(5), pytest.raises(cli.ContainerError, match="no answer in 5 s"):
        cli.containers()


def test_the_calls_after_a_signal_run_in_a_process_group_of_their_own(fake_container,
                                                                      monkeypatch):
    groups = []
    real = cli.subprocess.Popen

    def popen(argv, **kw):
        groups.append((argv[1], kw.get("process_group"), kw.get("stdin")))
        return real(argv, **kw)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    cli.containers()
    cli.containers(own_group=True)
    cli.stop("gmlx-pi-1", timeout=5)
    cli.kill("gmlx-pi-1", signal="SIGINT")
    cli.kill("gmlx-pi-1")
    cli.delete("gmlx-pi-1")
    cli.hangup_copy("gmlx-pi-1", "/opt/gmlx/gmlx-entry", "0f3a")
    assert groups == [("ls", None, subprocess.DEVNULL)] + [
        (call, 0, subprocess.DEVNULL) for call in ("ls", "stop", "kill", "kill", "delete", "exec")]
    assert fake_container.calls("exec") == [
        ["exec", "gmlx-pi-1", "/opt/gmlx/gmlx-entry", "--hangup", "0f3a"]]


def test_stop_follows_the_query_timeout(fake_container, monkeypatch):
    seen = []
    real = cli.subprocess.run

    def run(argv, **kw):
        seen.append(kw.get("timeout"))
        return real(argv, **kw)
    monkeypatch.setattr(cli.subprocess, "run", run)
    cli.stop("gmlx-pi-1", timeout=5)
    with cli.query_timeout(5):
        cli.stop("gmlx-pi-1", timeout=5)
    assert seen == [5 + cli.QUERY_TIMEOUT, 10]


def test_builder_notice_without_settle_leaves_an_owed_builder(fake_container,
                                                             no_other_builds, monkeypatch):
    settle = images._settle_builder
    monkeypatch.setattr(images, "_settle_builder", lambda say: None)   # killed first
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    monkeypatch.setattr(images, "_settle_builder", settle)
    assert images.builder_notice(settle=False) is None      # this launch builds next
    assert _stops(fake_container) == 0 and images._owed_path().exists()
    assert images.builder_notice() is None                   # settled now
    assert _stops(fake_container) == 1 and not images._owed_path().exists()


def test_a_failed_owed_record_never_hides_the_build_error(fake_container, no_other_builds,
                                                         monkeypatch):
    lines = []

    def refuse(path, value):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(images, "_write_date", refuse)
    fake_container.update(fail_build=True)
    with pytest.raises(cli.ContainerError):
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=lines.append)
    assert any("could not record that launch started the image builder" in line
               for line in lines)
    fake_container.update(fail_build=False, builder=False)      # stopped again
    lines.clear()
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=lines.append)
    assert sum("could not record" in line for line in lines) == 1


def test_date_records_use_a_temporary_name_per_process(fake_container, monkeypatch):
    names = []
    real = images.os.replace

    def replace(src, dst):
        names.append(os.path.basename(src))
        return real(src, dst)
    monkeypatch.setattr(images.os, "replace", replace)
    images.images_dir().mkdir(parents=True, exist_ok=True)
    images._write_date(images._owed_path(), "2026-09-28T09:00:00Z")
    assert len(names) == 1 and names[0].startswith(f".builder-owed.{os.getpid()}.")
    assert images._read_date(images._owed_path()) == "2026-09-28T09:00:00Z"
    assert sorted(p.name for p in images.images_dir().glob("*builder-owed*")) == ["builder-owed"]


def test_a_date_record_that_cannot_be_written_leaves_no_temporary_file(
        fake_container, monkeypatch):
    def disk_full(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(images.os, "replace", disk_full)
    images.images_dir().mkdir(parents=True, exist_ok=True)
    with pytest.raises(OSError):
        images._write_date(images._owed_path(), "2026-09-28T09:00:00Z")
    assert list(images.images_dir().glob("*builder-owed*")) == []


def test_the_check_container_carries_the_launch_labels(fake_container):
    cli.run_entry_check("img", "/rt", "a")
    run = fake_container.calls("run")[0]
    labels = [run[i + 1] for i, arg in enumerate(run) if arg == "--label"]
    assert labels == ["gmlx.launch=1", f"gmlx.launch.pid={os.getpid()}"]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_interrupted_check_removes_its_container(fake_container, monkeypatch, error):
    real = cli._run

    def run(args, **kw):
        if args[0] == "run":
            raise error
        return real(args, **kw)
    monkeypatch.setattr(cli, "_run", run)
    with pytest.raises(error):
        cli.run_entry_check("img", "/rt", "a")
    deleted = fake_container.calls("delete")
    assert len(deleted) == 1 and deleted[0][-1].startswith("gmlx-check-")


def test_the_check_line_keeps_a_carriage_return(fake_container):
    line = '/start.sh names "/bin/sh\\r" in its #! line, which is not in the image.'
    raw = "/start.sh names /bin/sh\r in its #! line, which is not in the image."
    # The entry's prefix is dropped, since launch adds its own.
    fake_container.update(checks={"a": [126, f"[launch] {line}"], "b": [126, f"[launch] {raw}"]})
    assert cli.run_entry_check("img", "/rt", "a") == (126, line)
    assert cli.run_entry_check("img", "/rt", "b") == (126, raw)


# Agents: images keyed by launch target, the runtime stage shared

def _agent(**kw):
    from gmlx.config import LaunchAgentCfg
    return LaunchAgentCfg(**{"command": ["bot"], **kw})


def test_base_refs_in_accepts_the_runtime_base_and_refuses_agent_repositories():
    assert images.base_refs_in("FROM gmlx.invalid/launch-runtime-python:base\n") == [
        "runtime-python"]
    for bad in ("FROM gmlx.invalid/launch-agent-bot:base\n",
                "FROM gmlx.invalid/launch-agent-bot-build:abc\n"):
        with pytest.raises(images.ImageError, match="Name the base of a client or of a runtime"):
            images.base_refs_in(bad)


def test_an_agent_resolves_to_the_runtime_stage_its_own_image_or_nothing(tmp_path):
    cfg = _config(_agents={
        "bot": _agent(runtime="python"),
        "img": _agent(image="ghcr.io/x/bot"),
        "mix": _agent(runtime="python", image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim")})
    plan = images.resolve_image("agent-bot", cfg.for_target("agent-bot"), cfg.container,
                                stage=images.stage_for(cfg, "agent-bot"))
    assert (plan.kind, plan.client, plan.packages) == ("shipped", "runtime-python", [])
    plan = images.resolve_image("agent-img", cfg.for_target("agent-img"), cfg.container,
                                stage=images.stage_for(cfg, "agent-img"))
    assert (plan.kind, plan.client, plan.ref) == ("image", "agent-img", "ghcr.io/x/bot")
    plan = images.resolve_image("agent-mix", cfg.for_target("agent-mix"), cfg.container,
                                stage=images.stage_for(cfg, "agent-mix"))
    assert plan.kind == "image"
    assert images.stage_for(cfg, "pi") == "pi" and images.stage_for(cfg, "agent-img") is None
    with pytest.raises(images.ImageError, match="bot names no image to run. Set "
                                               "launch.agents.bot.image, launch.agents.bot.build "
                                               "or launch.agents.bot.runtime"):
        images.resolve_image("agent-bot", LaunchClientCfg(), cfg.container)


def test_agent_messages_use_the_agent_config_path_and_name(tmp_path):
    with pytest.raises(images.ImageError, match=r"^launch\.agents\.bot\.build is 'ctx'"):
        images.resolve_image("agent-bot", LaunchClientCfg(build="ctx"), LaunchContainerCfg())
    with pytest.raises(images.ImageError, match=r"^launch\.agents\.bot\.build names .*/no, which"):
        images.resolve_image("agent-bot", LaunchClientCfg(build=str(tmp_path / "no")),
                             LaunchContainerCfg())
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM debian\n")
    with pytest.raises(images.ImageError, match="could change the bot build: folder"):
        images.resolve_image("agent-bot", LaunchClientCfg(build=str(ctx)), LaunchContainerCfg(),
                             writable=[str(tmp_path)])
    plan = images.resolve_image("agent-bot", LaunchClientCfg(build=str(ctx)),
                                LaunchContainerCfg())
    assert images._shipped_build_failure("runtime-python", [], 1).startswith(
        "the build of the Python runtime image failed")
    assert images._shipped_build_failure("agent-bot", [], 1).startswith(
        "the build of the bot image failed")
    assert plan.client == "agent-bot"
    assert images.build_repo("agent-bot") == "gmlx.invalid/launch-agent-bot-build"
    info = cli.ImageInfo(name="x", digest=D1, architectures=["linux/arm64"], arm64=True)
    ready = images.ReadyImage("image", "x", info, f"x@{D1}", "found", "agent-bot")
    with pytest.raises(images.ImageError, match="Set launch.agents.bot.command to the command"):
        images.image_command(ready, "image", [], [])


def test_a_runtime_base_in_an_agent_build_installs_no_packages(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM gmlx.invalid/launch-runtime-python:base\n"
                                       "FROM gmlx.invalid/launch-omp:base\n")
    cfg = _config(omp={"packages": ["jq"]})
    plan = images.resolve_image("agent-bot", LaunchClientCfg(build=str(ctx)), cfg.container)
    assert plan.bases == ["runtime-python", "omp"]
    assert plan.base_packages == {"runtime-python": [], "omp": ["jq"]}


def test_named_records_and_cleans_a_removed_agents_build_repository(fake_container, tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Containerfile").write_text("FROM debian\n")
    with_agent = _config(_agents={"bot": _agent(build=str(ctx)),
                                  "img": _agent(image="me/box:1")})
    assert images._named(with_agent) == {"images": {"agent-img": "me/box:1"},
                                         "builds": ["agent-bot"]}
    plan = images.resolve_image("agent-bot", with_agent.for_target("agent-bot"),
                                with_agent.container)
    images.forget_unnamed(with_agent, _quiet)
    ready = images.ensure_image(plan, say=_quiet)
    assert ready.tag.startswith("gmlx.invalid/launch-agent-bot-build:")
    images.forget_unnamed(with_agent, _quiet)
    assert ready.tag in fake_container.load()["images"]
    images.forget_unnamed(_config(), _quiet)                  # the agent is removed
    store = fake_container.load()["images"]
    assert not any(name.startswith("gmlx.invalid/launch-agent-bot-build") for name in store)


def test_two_agents_share_the_runtime_image_and_one_keeps_it_used(fake_container):
    tag, base = images.shipped_tag("runtime-python", []), images.base_ref("runtime-python")
    names = [tag, base, "gmlx.invalid/launch-agent-bot-build:x"]
    two = _config(_agents={"bot": _agent(runtime="python"), "cat": _agent(runtime="python")})
    one = _config(_agents={"cat": _agent(runtime="python")})
    none = _config()
    assert {tag, base} <= images._used_names(two, names)
    assert {tag, base} <= images._used_names(one, names)
    assert not {tag, base} & images._used_names(none, names)
    # An agent's build repository counts as used while it sets build:, and
    # the runtime image stays used when its build folder cannot be read.
    broken = _config(_agents={"bot": _agent(runtime="python", build="/no/such/folder")})
    assert set(names) <= images._used_names(broken, names)
    fake_container.update(images={tag: _img(D1, size=1 << 30), base: _img(D1, size=1 << 30)})
    assert images.disk_report(two)[2] == []
    assert images.disk_report(none)[2] == sorted([tag, base])


def test_the_check_is_skipped_for_uv_on_the_runtime_image(fake_container):
    info = cli.ImageInfo(name="x", digest=D1, architectures=["linux/arm64"], arm64=True)
    ready = images.ReadyImage("shipped", "x", info, f"x@{D1}", "found", "runtime-python")
    images.check_command(ready, "uv", "/rt", shell=False, say=_quiet)
    assert not fake_container.calls("run")
    assert set(images.CLIENT_BINARY) == set(LAUNCH_CLIENTS)
    assert set(images.RUNTIME_BINARY) == set(images.RUNTIME_STAGES.values())


def test_no_container_call_reads_stdin(fake_container, monkeypatch):
    """A launch with no terminal, such as one that --detach starts or a
    script that pipes input to an agent, never waits in a container call,
    and no call takes the agent's input. Launch asks the kernel question
    itself, so the service start and the kernel download read nothing."""
    seen: dict[str, object] = {}

    class Popen(subprocess.Popen):
        # subprocess.run starts its process through Popen too.
        def __init__(self, argv, **kw):
            seen[" ".join(argv[1:3])] = kw.get("stdin")
            super().__init__(argv, **kw)
    monkeypatch.setattr(cli.subprocess, "Popen", Popen)
    fake_container.update(registry={"debian:12": {"digest": D1}})
    cli.system_start(kernel=False)
    cli.pull("debian:12")
    cli._run_watched(["build", "--file", "/ctx/Containerfile", "/ctx"])
    cli.containers()
    assert seen == {"system start": subprocess.DEVNULL, "image pull": subprocess.DEVNULL,
                    "build --file": subprocess.DEVNULL, "ls --all": subprocess.DEVNULL}
    seen.clear()
    cli.system_start(kernel=True)
    cli.kernel_set_recommended()
    assert seen == {"system start": subprocess.DEVNULL, "system kernel": subprocess.DEVNULL}
