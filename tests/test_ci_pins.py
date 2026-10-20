#!/usr/bin/env python3
"""The CI workflow pre-installs some dependencies by exact version to get a
wheel instead of an sdist build. A pin below the pyproject floor is worse than
no pin: the editable install upgrades it, so the job tests a version nobody
chose. This anchors the workflow pins against pyproject, and the mlx and
mlx-kquant pins against the mlx that the copied mlx-vlm needs. It also
checks that the test job installs the all extra, that the drift step
fails when its report fails, and the job conditions and permissions the
workflows rely on.
"""
from __future__ import annotations

import re
import shlex
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"
_INSTALL = re.compile(r"pip install\s+(.+)")


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _declared() -> dict[str, Requirement]:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    out = {}
    for spec in deps:
        r = Requirement(spec)
        out[_canonical(r.name)] = r
    return out


def _ci_pins() -> dict[str, Requirement]:
    """Every version-constrained requirement a `pip install` line in the
    workflow names, quoted or not. Flags, paths, and unconstrained names
    are skipped."""
    out = {}
    for line in _WORKFLOW.read_text().splitlines():
        m = _INSTALL.search(line)
        if m is None:
            continue
        for tok in shlex.split(m.group(1)):
            if tok.startswith("-"):
                continue
            try:
                r = Requirement(tok)
            except InvalidRequirement:
                continue
            if str(r.specifier):
                out[_canonical(r.name)] = r
    return out


def test_workflow_pins_found():
    """A rename of the install step must not turn this file into a no-op."""
    pins = _ci_pins()
    assert pins, f"no version-pinned pip install lines found in {_WORKFLOW}"
    assert "mlx-kquant" in pins, pins


@pytest.mark.parametrize("name", sorted(_ci_pins()))
def test_ci_pin_satisfies_pyproject(name):
    """Each exact CI pin must satisfy the pyproject constraint for that
    package. A wildcard pin checks its lower bound."""
    pin = _ci_pins()[name]
    declared = _declared().get(name)
    if declared is None or not str(declared.specifier):
        pytest.skip(f"{name} is unconstrained in pyproject")
    versions = [s.version for s in pin.specifier
                if s.operator in ("==", "===")]
    if not versions:
        pytest.skip(f"{name} pre-install is not an exact pin: {pin.specifier}")
    for v in versions:
        probe = v[:-2] + ".0" if v.endswith(".*") else v
        assert declared.specifier.contains(probe, prereleases=True), (
            f"CI pins {name} {pin.specifier}, which does not satisfy the "
            f"pyproject constraint {declared.specifier}. The editable "
            "install would silently upgrade it; bump the workflow pin.")


def _exact(req: Requirement) -> str | None:
    versions = [s.version for s in req.specifier if s.operator == "=="]
    return versions[0] if len(versions) == 1 else None


def _local_requirements(name: str, version: str | None):
    """The installed distribution's requirements, or None when it is
    missing or is not `version`."""
    try:
        if version is None or metadata.version(name) != version:
            return None
        reqs = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return None
    return [Requirement(r) for r in reqs]


def _mlx_specifier(reqs) -> SpecifierSet:
    """The mlx constraint a plain install (no extras) applies."""
    spec = SpecifierSet()
    for r in reqs:
        if _canonical(r.name) != "mlx":
            continue
        if r.marker is None or r.marker.evaluate({"extra": ""}):
            spec &= r.specifier
    return spec


def _copy_mlx_floor() -> str:
    """The highest mlx floor the copied releases declare, as the manifest
    recorded their requirements."""
    import json

    manifest = json.loads(
        (_ROOT / "gmlx" / "_vendor" / "manifest.json").read_text())
    reqs = [Requirement(r) for p in manifest["packages"].values()
            for r in p["requires"]]
    floors = [s.version for s in _mlx_specifier(reqs)
              if s.operator in (">=", "==", "~=")]
    assert floors, "the copied releases declare no mlx floor"
    return max(floors, key=Version)


def test_kquant_pin_admits_the_copy_mlx_floor():
    """An mlx-kquant release pins mlx exactly. The CI pin must name one whose
    mlx meets the floor the copied mlx-vlm declares, or gmlx refuses to run
    on it. Read from the local install; skipped when the local mlx-kquant
    is not the pinned version."""
    kq = _local_requirements("mlx-kquant", _exact(_ci_pins()["mlx-kquant"]))
    if kq is None:
        pytest.skip("the local mlx-kquant is not the pinned version")
    floor = _copy_mlx_floor()
    kq_mlx = _mlx_specifier(kq)
    pins = [s.version for s in kq_mlx if s.operator == "=="]
    assert pins and all(Version(p) >= Version(floor) for p in pins), (
        f"the pinned mlx-kquant requires mlx{kq_mlx}, below mlx {floor}, the "
        "floor of the copied mlx-vlm. Bump the mlx-kquant pin to a release "
        "built against that mlx or a later one.")


def test_ci_mlx_pin_meets_the_copy_floor():
    pin = _exact(_ci_pins()["mlx"])
    assert pin is not None, _ci_pins()["mlx"]
    assert Version(pin) >= Version(_copy_mlx_floor()), pin


def _run(job: str, step: str) -> str:
    """The run text of the step named ``step`` in workflow job ``job``."""
    import yaml

    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"][job]["steps"]
    return next(s["run"] for s in steps if s.get("name") == step)


def test_the_test_job_installs_the_all_extra():
    """The chat, voice and assistant tests skip themselves when their extra
    is absent, so the test job installs ``all`` for them to run."""
    toks = [tok for m in _INSTALL.finditer(_run("test", "Install"))
            for tok in shlex.split(m.group(1))]
    assert ".[all]" in toks, toks


def test_the_drift_step_fails_when_the_report_fails():
    """The report is piped through tee. Without pipefail set before the
    pipe, the step ends with the status of tee."""
    run = _run("drift", "Drift report")
    assert "set -o pipefail" in run, run
    assert run.index("set -o pipefail") < run.index("| tee"), run


_WORKFLOWS = _ROOT / ".github" / "workflows"


def _yaml(name: str) -> dict:
    import yaml

    return yaml.safe_load((_WORKFLOWS / name).read_text())


def test_the_install_step_upgrades_pip_before_the_group_install():
    """``pip install --group`` needs pip 25.1, newer than setup-python's on 3.11."""
    run = _run("test", "Install")
    assert run.index("pip install --upgrade pip") < run.index("--group test"), run


@pytest.mark.parametrize("name", ["test.yml", "gate.yml"])
def test_a_job_that_names_permissions_keeps_contents_read(name):
    """A job's permissions block drops each permission it does not name, and
    checkout and every ref fetch need contents: read."""
    jobs = _yaml(name)["jobs"]
    bad = [j for j, job in jobs.items()
           if "permissions" in job and job["permissions"].get("contents") != "read"]
    assert not bad, bad


def test_each_schedule_condition_names_a_declared_cron():
    """The file has a weekly and a daily cron. A condition on the event name
    alone would run a weekly job every day."""
    text = _WORKFLOW.read_text()
    on = _yaml("test.yml")
    crons = {c["cron"] for c in (on.get("on") or on[True])["schedule"]}
    assert "event_name == 'schedule'" not in text
    assert "github.event_name }}\" = schedule" not in text
    named = set(re.findall(r"github\.event\.schedule\s*==\s*'([^']+)'", text))
    named |= set(re.findall(r'github\.event\.schedule \}\}" = "([^"]+)"', text))
    assert named == crons, (named, crons)


def test_the_gate_status_posts_only_for_this_repository():
    """A fork's token cannot post, so its run skips the step instead of
    failing. A dispatch run has no pull_request payload, so the step lets
    it through and checks the head repository it looked up."""
    steps = _yaml("gate.yml")["jobs"]["gate-status"]["steps"]
    post = next(s for s in steps if s.get("name") == "Post")
    assert post["if"] == ("github.event_name == 'workflow_dispatch' || "
                          "github.event.pull_request.head.repo.full_name == github.repository")
    assert '[ "$HEAD_REPO" != "$GH_REPO" ]' in post["run"]


def test_the_gate_job_follows_the_test_legs_of_a_pull_request():
    """A red test leg still posts the gate status, and a push to main has no
    pull request to post on. A called workflow gets only what its caller
    grants, so the job grants what gate.yml needs."""
    job = _yaml("test.yml")["jobs"]["gate"]
    assert job["uses"] == "./.github/workflows/gate.yml"
    assert job["needs"] == ["test"]
    assert job["if"] == "${{ !cancelled() && github.event_name == 'pull_request' }}"
    assert job["permissions"] == {"contents": "read", "statuses": "write", "pull-requests": "read"}
    gate = _yaml("gate.yml")
    assert "workflow_call" in (gate.get("on") or gate[True])
