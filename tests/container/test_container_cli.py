"""gmlx/container/cli.py and images.py against a fake ``container`` command:
the wrapper's parsing, building and pulling images, the ``:base`` tag, digest
references, tag cleanup, the one-time command check and the rebuild hash."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gmlx.config import LaunchClientCfg, LaunchContainerCfg
from gmlx.container import cli, ignore, images
from gmlx.container.state import FileLock

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


def _img(digest=D1, **kw):
    return {"digest": digest, **kw}


def _quiet(_line):
    pass


# The wrapper

def test_version_and_status(fake_container):
    fake_container.update(version="1.4.1")
    assert cli.version() == (1, 4, 1)
    assert cli.system_running()
    fake_container.update(running=False)
    assert not cli.system_running()


def test_missing_binary_names_the_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cli.find() is None
    with pytest.raises(cli.ContainerError, match="brew install container") as e:
        cli.version()
    assert str(e.value).count("Apple container") == 1


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


def test_volume_create_passes_the_label_and_size(fake_container):
    cli.volume_create("pg", size="32G")
    assert fake_container.calls("volume", "create") == [
        ["volume", "create", "--label", "gmlx.launch=1", "-s", "32G", "pg"]]
    vol = cli.volume_list()[0]
    assert vol.name == "pg" and vol.labels == {"gmlx.launch": "1"}


def test_nothing_in_the_wrapper_deletes_a_volume():
    source = Path(cli.__file__).read_text() + Path(images.__file__).read_text()
    assert '"volume", "delete"' not in source and "volume delete" not in source


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
    with pytest.raises(images.ImageError, match="stop it with: container builder stop$"):
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


def test_other_build_failures_keep_their_message(fake_container, capsys):
    fake_container.update(fail_build="[ERROR] Could not resolve dependencies for project")
    with pytest.raises(cli.ContainerError, match=r"`container build --file /ctx/Containerfile` "
                                                 r"failed \(exit 1\)\.$"):
        cli.build("/ctx", file="/ctx/Containerfile", tags=["t"])
    assert "Could not resolve dependencies" in capsys.readouterr().err


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
        with pytest.raises(images.ImageError, match="stable base"):
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
    with pytest.raises(images.ImageError, match="refuses a Containerfile"):
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
    with pytest.raises(images.ImageError, match="shares read-write"):
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
        with pytest.raises(images.ImageError, match="private homes of the clients"):
            images.resolve_image("claude-code", LaunchClientCfg(build=str(build)), cfg)


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
    with pytest.raises(images.ImageError, match="an earlier launch shared read-write"):
        images.resolve_image("pi", LaunchClientCfg(build=str(ctx)), cfg)
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
    with pytest.raises(images.ImageError, match="shares read-write"):
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
    for client in LAUNCH_CLIENTS:
        assert f"    {client})" in text
    assert set(images.CLIENT_BINARY) == set(LAUNCH_CLIENTS)


def test_image_override_notes_unused_packages():
    plan = images.resolve_image("pi", LaunchClientCfg(packages=["make"]), LaunchContainerCfg(),
                                image_override="debian:12")
    assert plan.kind == "image" and "packages: list is not used" in plan.notices[0]


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


def test_missing_command_refuses_or_warns_under_shell(fake_container):
    ready = _ready(fake_container)
    msg = "gmlx-entry: claude is not on the image's PATH (/usr/bin)."
    fake_container.update(checks={"claude": [127, msg]})
    with pytest.raises(images.ImageError, match="not on the image's PATH"):
        images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    said = []
    images.check_command(ready, "claude", "/rt", shell=True, say=said.append)
    assert said == [f"[launch] warning: {msg}"]


def test_shipped_images_skip_the_check_only_for_their_own_client(fake_container):
    ready = _ready(fake_container, "shipped")
    images.check_command(ready, "claude", "/rt", shell=False, say=_quiet)
    assert fake_container.calls("run") == []
    # A command: list on the shipped image names a command the image may lack.
    fake_container.update(checks={"start.sh": [127, "gmlx-entry: start.sh is not there."]})
    with pytest.raises(images.ImageError, match="start.sh is not there"):
        images.check_command(ready, "start.sh", "/rt", shell=False, say=_quiet)


def test_a_command_without_the_execute_bit_refuses_like_a_missing_one(fake_container):
    ready = _ready(fake_container)
    msg = "gmlx-entry: /usr/local/bin/start.sh has no execute bit. Run chmod 755 on it."
    fake_container.update(checks={"start.sh": [126, msg]})
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
    fake_container.update(checks={"x": [125, "gmlx-entry: cannot listen."]})
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
    with pytest.raises(cli.ContainerError, match="gave no answer"):
        images.check_command(ready, "other", "/rt", shell=False, say=_quiet)
    (delete,) = fake_container.calls("delete")
    assert delete[:2] == ["delete", "--force"] and delete[2].startswith("gmlx-check-")


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


def test_describe_and_age_note(fake_container):
    fake_container.update(images={"x:1": _img(created="2026-08-01T00:00:00Z")})
    ready = images.ReadyImage("shipped", "x:1", cli.image_info("x:1"), "x@" + D1, "found")
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    assert images.describe(ready, now) == "[launch] image x:1, built 57 days ago"
    assert "57 days old" in images.image_age_note(ready, now)
    assert images.image_age_note(ready, datetime(2026, 8, 20, tzinfo=timezone.utc)) is None


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


def _wait_for(said, text):
    deadline = time.monotonic() + 10
    while not any(text in line for line in said):
        assert time.monotonic() < deadline, f"no {text!r} in {said}"
        time.sleep(0.01)


def test_second_launch_waits_for_the_build_and_builds_nothing(fake_container):
    tag = images.shipped_tag("pi", [])
    held = images.repo_lock("gmlx.invalid/launch-pi")
    said: list[str] = []
    t, out = _in_thread(lambda: images.ensure_image(images.ImagePlan("shipped", "pi"),
                                                    say=said.append))
    _wait_for(said, "waiting for another launch")
    fake_container.update(images={tag: _img(D1), "gmlx.invalid/launch-pi:base": _img(D1)})
    held.release()
    t.join(10)
    assert out["value"].action == "found"
    assert "builds" not in fake_container.load()


def test_base_rebuild_waits_while_a_user_build_holds_it_shared(fake_container):
    shared = images.repo_lock("gmlx.invalid/launch-pi", shared=True)
    said: list[str] = []
    t, out = _in_thread(lambda: images.ensure_image(images.ImagePlan("shipped", "pi"),
                                                    rebuild=True, say=said.append))
    _wait_for(said, "waiting for another launch")
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


def test_records_drop_dead_pids_and_gone_images(fake_container):
    first = images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    gone = "gmlx.invalid/launch-pi@" + D2

    def plant(records):
        records[gone] = {"clients": ["omp"], "pids": []}
        records[first.run_ref]["pids"] = [999999, os.getpid()]
    images._update_records(plant)
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
    assert builds[0] == ("[launch] step 2 of 3: building the pi image, which first downloads "
                         "about 80 MB for the node:22-bookworm-slim base image")
    assert builds[1].startswith("[launch] building ")
    assert sum("step 2 of 3" in line for line in said) == 1


def test_the_node_download_is_named_only_when_it_happens(fake_container):
    fake_container.update(images={images._node_base(): _img(D2)})
    said = []
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=said.append)
    assert "[launch] building the pi image" in said


def test_the_shipped_image_installs_only_pinned_versions():
    """The base by digest, each client by version, each download by version
    and sha256, so a build installs what gmlx names."""
    import re
    text = images.SHIPPED_CONTAINERFILE.read_text()
    assert re.fullmatch(r"docker\.io/library/node:22-bookworm-slim@sha256:[0-9a-f]{64}",
                        images._node_base())
    assert "releases/latest" not in text and "url_effective" not in text
    for line in re.findall(r"npm install -g [^;]*", text):
        for pkg in line.split()[3:]:
            assert re.search(r".@\d[\w.-]*$", pkg), pkg
    for line in re.findall(r"pip install --no-cache-dir [^;]*", text):
        pkgs = [w for w in line.split()[3:] if not w.startswith(("-", "http", "\\"))]
        assert pkgs and all(re.fullmatch(r"[\w-]+==[\w.]+", w) for w in pkgs), line
    for url in re.findall(r'"\$gh/[^"]+"', text):
        assert "/releases/download/v" in url, url
    fetches = re.findall(r"fetch (\S+) \\?\s*\n?\s*\"([^\"]+)\" \\\s*\n\s*([0-9a-f]{64})", text)
    assert len(fetches) == 3


def test_the_shipped_layers_share_the_common_packages():
    text = images.SHIPPED_CONTAINERFILE.read_text()
    first_run = text.index("RUN ")
    assert text.index("ARG CLIENT") > first_run
    assert text.index("ARG EXTRA_PACKAGES") > text.index("npm install -g opencode-ai")
    assert "$EXTRA_PACKAGES" not in text[:text.index("ARG EXTRA_PACKAGES")]


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
    assert len(names) == 1 and names[0].startswith(f"builder-owed.{os.getpid()}.")
    assert images._read_date(images._owed_path()) == "2026-09-28T09:00:00Z"
    assert sorted(p.name for p in images.images_dir().glob("builder-owed*")) == ["builder-owed"]


def test_a_date_record_that_cannot_be_written_leaves_no_temporary_file(
        fake_container, monkeypatch):
    def disk_full(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(images.os, "replace", disk_full)
    images.images_dir().mkdir(parents=True, exist_ok=True)
    with pytest.raises(OSError):
        images._write_date(images._owed_path(), "2026-09-28T09:00:00Z")
    assert list(images.images_dir().glob("builder-owed*")) == []


def test_the_check_line_keeps_a_carriage_return(fake_container):
    line = 'gmlx-entry: /start.sh names "/bin/sh\\r" in its #! line, which is not in the image.'
    raw = "gmlx-entry: /start.sh names /bin/sh\r in its #! line, which is not in the image."
    fake_container.update(checks={"a": [126, line], "b": [126, raw]})
    assert cli.run_entry_check("img", "/rt", "a") == (126, line)
    assert cli.run_entry_check("img", "/rt", "b") == (126, raw)
