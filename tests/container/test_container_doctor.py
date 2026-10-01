"""gmlx doctor's container row: the Apple container install, the service, the
guest entry, file handles, and what launch keeps on disk."""

from __future__ import annotations

import os

import pytest

import gmlx.commands.doctor as doctor
from gmlx.container import runtime


@pytest.fixture
def box(fake_container, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    entry = tmp_path / "gmlx-entry"
    entry.write_bytes(b"\x7fELF-fake")
    monkeypatch.setattr(runtime, "entry_path", lambda: entry)
    counts = {"kern.num_files": 1000, "kern.maxfiles": 491520,
              "kern.maxfilesperproc": 245760}
    monkeypatch.setattr(doctor, "_sysctl_int", lambda name: counts.get(name))
    fake_container.home, fake_container.entry, fake_container.counts = home, entry, counts
    return fake_container


def _enable(home):
    cfg = home / ".config" / "gmlx" / "gmlx.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("launch:\n  container:\n    enabled: true\n")


def test_skips_when_neither_installed_nor_configured(box, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    row = doctor.check_container()
    assert row["status"] == "SKIP" and "(brew install container)" in row["detail"]


def test_fails_when_enabled_but_not_installed(box, monkeypatch):
    _enable(box.home)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    row = doctor.check_container()
    assert row["status"] == "FAIL"
    assert row["detail"] == ("container mode is on, but Apple container is not installed "
                             "(brew install container)")


def test_absent_off_macos(box, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    assert doctor.check_container() is None


def test_passes_with_the_version_and_file_handles(box):
    row = doctor.check_container()
    assert row["status"] == "PASS"
    assert row["detail"] == "container 1.4.1; 1,000 of 491,520 open files (245,760 per process)"


@pytest.mark.parametrize("enabled,status", [(True, "FAIL"), (False, "WARN")])
def test_missing_entry_fails_only_when_enabled(box, enabled, status):
    if enabled:
        _enable(box.home)
    box.entry.unlink()
    row = doctor.check_container()
    assert row["status"] == status
    assert "scripts/build_guest_entry.py" in row["detail"]


def test_a_stopped_service_is_only_information_when_container_mode_is_off(box):
    box.update(running=False)
    row = doctor.check_container()
    assert row["status"] == "PASS" and "container system start" in row["detail"]


def test_the_private_home_walk_is_capped(box, monkeypatch):
    from gmlx.container import settings
    home = settings.private_home("pi", "proj-1234abcd")
    for n in range(5):
        (home / f"f{n}").write_bytes(b"x" * 5000)
    monkeypatch.setattr(doctor, "_WALK_CAP", 20)
    monkeypatch.setattr(doctor, "_HOMES_LISTED", 10)
    [row] = doctor.check_homes()
    assert ", at least " in row["detail"]


def test_each_private_home_gets_a_row_with_its_folder_size_and_last_use(box):
    import json
    import time

    from gmlx.container import settings
    used = int(time.mktime((2026, 9, 28, 12, 0, 0, 0, 0, -1)))
    for client, project, folder, size in (("claude-code", "app-1", str(box.home / "app"), 2),
                                          ("open-webui", "default", None, 1)):
        home = settings.private_home(client, project)
        (home / "data").write_bytes(b"y" * (size << 20))
        settings.project_record_path(client, project).write_text(
            json.dumps({"folder": folder, "used": used}))
    rows = doctor.check_homes()
    assert [r["name"] for r in rows] == ["home"] * 2
    assert {r["status"] for r in rows} == {"PASS"}
    details = [r["detail"] for r in rows]
    assert "claude-code: ~/app, 2M, last used 2026-09-28" in details
    assert "open-webui: default project, 1M, last used 2026-09-28" in details


def test_homes_past_the_listed_ones_share_a_row(box, monkeypatch):
    from gmlx.container import settings
    for n in range(4):
        settings.private_home("pi", f"p-{n}")
    monkeypatch.setattr(doctor, "_HOMES_LISTED", 2)
    rows = doctor.check_homes()
    assert len(rows) == 3
    assert rows[-1]["detail"].startswith("and 2 more private homes under ")


def test_no_home_rows_off_macos_or_without_homes(box, monkeypatch):
    assert doctor.check_homes() == []
    from gmlx.container import settings
    settings.private_home("pi", "p-1")
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    assert doctor.check_homes() == []


def test_the_private_home_walk_counts_folders(tmp_path, monkeypatch):
    """A guest can make many empty folders, which hold no files to count."""
    root = tmp_path / "home"
    for n in range(50):
        (root / f"d{n}").mkdir(parents=True)
    visited = []
    real = doctor.os.walk

    def walk(top, *a, **k):
        for entry in real(top, *a, **k):
            visited.append(entry[0])
            yield entry
    monkeypatch.setattr(doctor.os, "walk", walk)
    budget = [10]
    assert doctor._folder_bytes(root, budget) == 0
    assert budget == [0] and len(visited) <= 11


def test_old_version_and_stopped_service_warn(box):
    _enable(box.home)
    box.update(version="1.3.0", running=False)
    row = doctor.check_container()
    assert row["status"] == "WARN"
    assert "container 1.3.0 is older than 1.4.0" in row["detail"]
    assert "container system start" in row["detail"]
    assert not box.calls("ls")                   # no queries on a stopped service


def test_file_handles_warn_only_while_a_launch_container_runs(box):
    box.counts["kern.num_files"] = 300000
    assert doctor.check_container()["status"] == "PASS"
    box.update(containers=[{"name": "gmlx-pi-1", "labels": {
        "gmlx.launch": "1", "gmlx.launch.client": "pi", "gmlx.launch.pid": str(os.getpid())}}])
    row = doctor.check_container()
    assert row["status"] == "WARN" and "narrow its shares" in row["detail"]


def test_reports_volumes_homes_and_images(box, tmp_path):
    disk = tmp_path / "volume.img"
    with open(disk, "wb") as f:
        f.truncate(8 << 30)                      # sparse: 8G apparent size
        f.write(b"x" * (2 << 20))
    box.update(volumes=[
        {"name": "pg", "labels": {"gmlx.launch": "1"}, "source": str(disk)},
        {"name": "theirs", "labels": {}, "source": str(disk)}],
        images={"gmlx.invalid/launch-pi:abc": {"digest": "sha256:" + "1" * 64,
                                               "size": 3 << 30},
                "gmlx.invalid/launch-pi:base": {"digest": "sha256:" + "1" * 64,
                                                "size": 3 << 30},
                "debian:bookworm-slim": {"digest": "sha256:" + "2" * 64, "size": 1 << 30}})
    detail = doctor.check_container()["detail"]
    assert "volumes pg 2M" in detail and "theirs" not in detail   # allocated, not 8G
    assert "private homes" not in detail                          # rows of their own
    assert "1 launch image, 3G of layers" in detail


def test_names_images_no_setting_uses_with_the_delete_command(box):
    box.update(images={"gmlx.invalid/launch-pi-build:x": {"digest": "sha256:" + "4" * 64,
                                                          "size": 1 << 30}})
    detail = doctor.check_container()["detail"]
    assert "1 launch image, 1G of layers" in detail
    assert ("1 image reference that no setting uses (container image delete "
            "gmlx.invalid/launch-pi-build:x)") in detail


def test_leftover_containers_warn_with_memory_and_stop_command(box):
    box.update(containers=[
        {"name": "gmlx-omp-1", "memory": 4 << 30,
         "labels": {"gmlx.launch": "1", "gmlx.launch.client": "omp",
                    "gmlx.launch.pid": "999999"}},
        {"name": "gmlx-pi-2", "labels": {"gmlx.launch": "1", "gmlx.launch.client": "pi",
                                         "gmlx.launch.pid": str(os.getpid())}}])
    row = doctor.check_container()
    assert row["status"] == "WARN"
    assert "gmlx-omp-1 is left over, 4G (container stop gmlx-omp-1)" in row["detail"]
    assert "gmlx-pi-2" not in row["detail"]


def test_queries_time_out_quickly_and_an_idle_builder_is_reported(box, monkeypatch):
    from gmlx.container import cli
    seen = []
    real = cli.query_timeout

    def spy(seconds):
        seen.append(seconds)
        return real(seconds)
    monkeypatch.setattr(cli, "query_timeout", spy)
    from gmlx.container import images
    # Other test runs can start the fake `container build` at the same time.
    monkeypatch.setattr(images, "_other_builds", lambda: False)
    box.update(builder=True)
    row = doctor.check_container()
    assert seen == [doctor.DOCTOR_QUERY_TIMEOUT] == [5.0]
    # A builder launch did not start is information only.
    assert row["status"] == "PASS" and "container builder stop" in row["detail"]
    started = cli.builder().started
    images._owed_path().parent.mkdir(parents=True, exist_ok=True)
    images._owed_path().write_text(started + "\n")
    row = doctor.check_container()                   # a launch owes its stop
    assert row["status"] == "WARN" and "container builder stop" in row["detail"]
    assert cli.builder().state == "running"          # doctor never stops it


def test_a_service_that_does_not_answer_warns(box, monkeypatch):
    from gmlx.container import cli

    def hang(*a, **k):
        raise cli.ContainerError("container gave no answer in 5 s")
    monkeypatch.setattr(cli, "containers", hang)
    row = doctor.check_container()
    assert row["status"] == "WARN" and "no answer in 5 s" in row["detail"]
