"""gmlx/container/cli.py and images.py against a fake ``container`` command:
the wrapper's parsing, building and pulling images, the ``:base`` tag, digest
references, tag cleanup, the one-time command check and the rebuild hash."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gmlx.config import LaunchClientCfg, LaunchContainerCfg
from gmlx.container import cli, images
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
    with pytest.raises(cli.ContainerError, match="brew install container"):
        cli.version()


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


def test_a_builder_launch_started_is_stopped_after_the_build(fake_container):
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop")
    assert not fake_container.load()["builder"]


def test_a_builder_that_was_running_keeps_running(fake_container):
    fake_container.update(builder=True)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert not fake_container.calls("builder", "stop")


def test_the_builder_stays_while_another_launch_builds(fake_container):
    other = FileLock(images.images_dir() / "builder.lock", shared=True)
    try:
        images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
        assert not fake_container.calls("builder", "stop")
    finally:
        other.release()
    # The marker stays, so the launch that finishes last stops it.
    assert (images.images_dir() / "builder-started").exists()


def test_a_stale_marker_never_stops_a_builder_you_started(fake_container):
    # A launch killed mid-build left its marker, and you started the builder
    # afterwards.
    marker = images.images_dir() / "builder-started"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("999999")
    fake_container.update(builder=True, builder_started="2099-01-01T00:00:00Z")
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert not fake_container.calls("builder", "stop")
    assert not marker.exists()


def test_a_killed_launchs_builder_is_adopted_and_stopped(fake_container):
    # A killed launch left its marker, and the builder it started runs on.
    marker = images.images_dir() / "builder-started"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("999999")
    written = time.time() - 600
    os.utime(marker, (written, written))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(written + 5))
    fake_container.update(builder=True, builder_started=started)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert fake_container.calls("builder", "stop")
    assert not marker.exists()


def test_a_builder_started_before_the_marker_keeps_running(fake_container, monkeypatch):
    # The marker was written, but the builder that runs after the build
    # started earlier, so it is not the one the launch started.
    real = images.cli.build

    def build(context, **kw):
        real(context, **kw)
        fake_container.update(builder_started="2000-01-01T00:00:00Z")
    monkeypatch.setattr(images.cli, "build", build)
    images.ensure_image(images.ImagePlan("shipped", "pi"), say=_quiet)
    assert not fake_container.calls("builder", "stop")


def test_packages_change_the_hash_and_reach_the_build(fake_container):
    assert images.shipped_tag("pi", []) != images.shipped_tag("pi", ["make"])
    images.ensure_image(images.ImagePlan("shipped", "pi", packages=["make", "jq"]), say=_quiet)
    assert fake_container.load()["builds"][0]["build_args"]["EXTRA_PACKAGES"] == "make jq"


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
        records[run_ref]["pid"] = pid
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

def _ready(fake_container, kind="image"):
    fake_container.update(images={"x:1": _img()})
    return images.ReadyImage(kind, "x:1", cli.image_info("x:1"), "x@" + D1, "found")


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


def test_shipped_images_skip_the_check(fake_container):
    images.check_command(_ready(fake_container, "shipped"), "claude", "/rt", shell=False,
                         say=_quiet)
    assert fake_container.calls("run") == []


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
