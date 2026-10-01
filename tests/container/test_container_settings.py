"""gmlx/container/settings.py: what a session shares, the refusals and
warnings, the private home, and the server-config checks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

from gmlx.config import LaunchClientCfg, LaunchContainerCfg
from gmlx.container import settings
from gmlx.container.settings import Mount, SettingsError


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake $HOME with a project, and the launch folders under it."""
    h = tmp_path / "home"
    (h / "src" / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    for var in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return h


def _plan(home, client="pi", cfg=None, **kw):
    cfg = cfg or LaunchClientCfg()
    kw.setdefault("cwd", str(home / "src" / "proj"))
    return settings.resolve_plan(client, cfg, **kw)


# Refusals and sensitive paths

@pytest.mark.parametrize("rel", ["", ".ssh", ".ssh/keys", ".config", ".local/share/gmlx/launch/pi"])
def test_auto_share_refuses(home, rel):
    path = os.path.realpath(home / rel) if rel else os.path.realpath(home)
    os.makedirs(path, exist_ok=True)
    assert settings.auto_share_refusal(path) is not None


def test_auto_share_refuses_system_folders_and_home_ancestors(home):
    for path in ("/", "/Users", "/tmp", "/private/tmp", os.path.dirname(os.path.realpath(home))):
        assert settings.auto_share_refusal(path) is not None
    assert settings.auto_share_refusal(os.path.realpath(home / "src" / "proj")) is None


def test_auto_share_refuses_the_temporary_folders(home, monkeypatch, tmp_path):
    """They hold the temporary files of every program, and gmlx's fallback
    socket folders."""
    t = tmp_path / "T"
    (t / "work").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(t))
    for path in (t, t / "work"):
        why = settings.auto_share_refusal(os.path.realpath(path))
        assert why is not None and "temporary files" in why
    per_user = os.path.realpath(tempfile.gettempdir())
    for path in ("/private/var/folders", per_user, os.path.join(per_user, "work")):
        assert settings.auto_share_refusal(path) is not None, path
    with pytest.raises(SettingsError, match="Launch from a project folder"):
        _plan(home, cwd=str(t / "work"))
    # A project under /private/tmp stays allowed.
    (tmp_path / "proj").mkdir()
    assert settings.auto_share_refusal(os.path.realpath(tmp_path / "proj")) is None


def test_without_tmpdir_only_gmlx_folders_in_tmp_are_refused(home, monkeypatch, tmp_path):
    """With TMPDIR unset, programs keep temporary files in /tmp, and so do
    scratch projects. Launch refuses only the folders gmlx keeps there."""
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert os.path.realpath(tempfile.gettempdir()) == "/private/tmp"
    (tmp_path / "proj").mkdir()
    assert settings.auto_share_refusal(os.path.realpath(tmp_path / "proj")) is None
    for name in ("gmlx-sessions-127-0-0-1-8080", "gmlx-launch-pi-3fa9c1"):
        folder = os.path.join("/private/tmp", name)
        assert settings.auto_share_refusal(folder) == (
            f"is {folder}, which holds the session sockets of gmlx")
        assert settings.auto_share_refusal(folder + "/sub").startswith(f"lies in {folder}")


def test_sensitive_hits_cover_all_three_relations(home):
    ssh = os.path.realpath(home) + "/.ssh"
    assert settings.sensitive_hits(ssh) == [ssh]                   # is one
    assert ssh in settings.sensitive_hits(ssh + "/keys")           # lies inside one
    assert ssh in settings.sensitive_hits(os.path.realpath(home))  # contains one
    assert settings.sensitive_hits(os.path.realpath(home / "src")) == []


def test_refused_current_folder_names_the_way_out(home):
    with pytest.raises(SettingsError, match="--no-mount-cwd"):
        _plan(home, cwd=str(home))


# Mount specs and rules

@pytest.mark.parametrize("spec, parsed", [
    ("~/a", ("HOME/a", None, False)),
    ("~/a:ro", ("HOME/a", None, True)),
    ("/x:/data", ("/x", "/data", False)),
    ("/x:/data:ro", ("/x", "/data", True)),
    ("/x:/data:rw", ("/x", "/data", False)),
])
def test_parse_mount_spec(home, spec, parsed):
    source, target, ro = settings.parse_mount_spec(spec)
    assert (source.replace(str(home), "HOME"), target, ro) == parsed


@pytest.mark.parametrize("spec", ["/x:data", ":/x", "/a:/b:/c", "/a:/b:ro:ro"])
def test_parse_mount_spec_refuses(spec):
    with pytest.raises(SettingsError):
        settings.parse_mount_spec(spec)


def test_explicit_mount_rules(home):
    (home / "file.txt").write_text("x")
    with pytest.raises(SettingsError, match="does not exist"):
        _plan(home, cli_mounts=["~/missing"])
    with pytest.raises(SettingsError, match="not a folder"):
        _plan(home, cli_mounts=["~/file.txt"])
    (home / "a,b").mkdir()
    with pytest.raises(SettingsError, match="a comma or an equals sign"):
        _plan(home, cli_mounts=["~/a,b"])


def test_explicit_sensitive_mount_is_honored_with_a_warning(home):
    (home / ".ssh").mkdir()
    plan = _plan(home, cli_mounts=["~/.ssh:ro"])
    assert any(m.source.endswith("/.ssh") and m.readonly for m in plan.mounts)
    assert plan.warnings == ["[launch] warning: the share ~/.ssh holds credentials. The "
                             "client can read every file in it."]
    plan = _plan(home, cli_mounts=["~/.ssh"])
    assert plan.warnings == ["[launch] warning: the share ~/.ssh holds credentials. The "
                             "client can read and change every file in it."]


def test_a_refusal_names_the_path_once(home):
    with pytest.raises(SettingsError, match=r"^will not share the current folder ~, because "
                                            r"it is your home folder\. Launch from"):
        _plan(home, cwd=str(home))
    (home / ".ssh" / "keys").mkdir(parents=True)
    with pytest.raises(SettingsError, match=r"current folder ~/\.ssh/keys, because it lies in "
                                            r"~/\.ssh, which holds credentials"):
        _plan(home, cwd=str(home / ".ssh" / "keys"))
    with pytest.raises(SettingsError, match=r"^the share ~/missing does not exist\.$"):
        _plan(home, cli_mounts=["~/missing:/m:ro"])
    (home / "keys").symlink_to(home / ".ssh")
    with pytest.raises(SettingsError, match=r"^seed: will not copy ~/keys, because it leads "
                                            r"to ~/\.ssh, which holds credentials\.$"):
        settings.seed_home(settings.private_home("pi"), ["~/keys"])


def test_normalize_drops_duplicates_and_orders_by_depth():
    a = Mount("/h/a", "/w/a/b")
    out = settings.normalize_mounts([a, Mount("/h/p", "/w"), a, Mount("/h/q", "/w/a")])
    assert [m.target for m in out] == ["/w", "/w/a", "/w/a/b"]


def test_normalize_drops_a_duplicate_that_differs_only_in_its_note():
    auto = Mount("/h/proj", "/h/proj", note="working folder")
    out = settings.normalize_mounts([auto, Mount("/h/proj", "/h/proj")])
    assert out == [auto]


def test_normalize_stores_the_normalized_guest_path():
    out = settings.normalize_mounts([Mount("/h/a", "/data/"), Mount("/h/b", "/data//x/../y")])
    assert [m.target for m in out] == ["/data", "/data/y"]
    with pytest.raises(SettingsError, match="both use /data in the container"):
        settings.normalize_mounts([Mount("/h/a", "/data/"), Mount("/h/b", "/data")])


def test_a_read_only_mount_of_the_current_folder_replaces_the_default_share(home):
    proj = os.path.realpath(home / "src" / "proj")
    plan = _plan(home, cli_mounts=[f"{proj}:ro"])
    shares = [m for m in plan.mounts if m.kind == "share"]
    assert [(m.source, m.target, m.readonly) for m in shares] == [(proj, proj, True)]
    assert plan.workdir == proj and plan.cwd_shared


@pytest.mark.parametrize("target,why", [
    ("/", "/ in the container. Choose a folder below it."),
    ("/proc", "/proc, which Linux in the container provides."),
    ("/sys/x", "/sys/x, inside /sys, which Linux in the container provides."),
    ("/dev", "/dev, which Linux in the container provides."),
    ("/opt/gmlx", "/opt/gmlx, where launch keeps its own program."),
    ("/var/host-services/x", "/var/host-services/x, inside /var/host-services, where launch "
                             "puts the sockets that reach the Mac."),
    ("/opt", "/opt, which would cover /opt/gmlx, where launch keeps its own program."),
    ("/var", "/var, which would cover /var/host-services, where launch puts the sockets "
             "that reach the Mac.")])
def test_reserved_targets_are_refused(target, why):
    with pytest.raises(SettingsError) as e:
        settings.normalize_mounts([Mount("/h/a", target)])
    if target != "/":
        why += " Choose another path in the container."
    assert str(e.value) == f"/h/a cannot use {why}"


def test_two_mounts_at_one_target_are_refused(home):
    (home / "other").mkdir()
    proj = os.path.realpath(home / "src" / "proj")
    with pytest.raises(SettingsError, match="both use"):
        _plan(home, cli_mounts=[f"~/other:{proj}"])


# mount_cwd order and the working folder

def test_mount_cwd_order(home):
    def shared(client, flag=None, global_=None, own=None):
        cfg = LaunchContainerCfg(mount_cwd=global_,
                                 clients={client: LaunchClientCfg(mount_cwd=own)})
        return _plan(home, client, cfg.for_client(client), mount_cwd=flag).cwd_shared
    assert shared("pi") and not shared("elia") and not shared("open-webui")
    assert shared("elia", global_=True)
    assert not shared("pi", global_=True, own=False)
    assert shared("elia", flag=True, own=False)
    assert not shared("pi", flag=False)


def test_workdir_is_the_private_home_without_a_share(home):
    plan = _plan(home, "elia")
    assert plan.workdir == str(plan.home) and not plan.cwd_shared
    assert any(m.kind == "home" and m.target == str(plan.home) for m in plan.mounts)
    assert plan.home == (home / ".local" / "share" / "gmlx" / "launch" / "elia" / "projects"
                         / "default" / "home")
    assert oct(plan.home.stat().st_mode & 0o777) == "0o700"


def test_guest_path_maps_through_a_share_at_another_path(home):
    data = home / "datasets"
    (data / "raw").mkdir(parents=True)
    plan = _plan(home, cli_mounts=["~/datasets:/data"], mount_cwd=False,
                 cwd=str(data / "raw"))
    assert plan.workdir == "/data/raw"


def test_guest_path_matches_whole_components_and_the_longest_share():
    shares = [Mount("/u/src/a", "/u/src/a"), Mount("/u/src/a/ro", "/inner", readonly=True)]
    assert settings.guest_path("/u/src/ab", shares) is None
    assert settings.guest_path("/u/src/a/x", shares) == "/u/src/a/x"
    assert settings.guest_path("/u/src/a/ro/y", shares) == "/inner/y"


# Projects and their private homes

def test_a_project_id_names_and_hashes_the_folder():
    import hashlib
    digest = hashlib.sha256(b"/Users/u/src/my app").hexdigest()[:8]
    assert settings.project_id("/Users/u/src/my app") == f"my_app-{digest}"
    assert settings.project_id(None) == "default"
    long = settings.project_id("/u/" + "x" * 50)
    assert long.startswith("x" * 32 + "-") and len(long) == 41
    assert settings.project_id("/u/\u9879\u76ee").startswith("__-")
    assert settings.project_id("/u/a") != settings.project_id("/v/a")


def test_each_project_gets_its_own_home_and_says_when_it_is_new(home):
    first = _plan(home, project="proj-1")
    assert first.new_home and first.project == "proj-1"
    assert first.home == settings.private_home_path("pi", "proj-1")
    assert not _plan(home, project="proj-1").new_home
    other = _plan(home, project="proj-2")
    assert other.new_home and other.home != first.home


def test_client_volumes_get_the_project_name(home):
    cfg = LaunchClientCfg(volumes=["pg:/var/lib/postgresql", "cache:/root/.cache"])
    plan = _plan(home, cfg=cfg, project="proj-1", project_volumes=["pg:/var/lib/postgresql"])
    names = {m.target: m.source for m in plan.volumes}
    assert names["/root/.cache"] == "cache"
    assert names["/var/lib/postgresql"] == settings.project_volume_name("pg", "proj-1")
    assert re.fullmatch(r"pg-[0-9a-f]{8}", names["/var/lib/postgresql"])
    assert len(settings.project_volume_name("v" * 300, "proj-1")) == 255


def test_the_project_record_keeps_the_folder_and_the_use(home):
    settings.write_project_record("pi", "proj-1", "/u/src/app")
    doc = settings.read_project_record("pi", "proj-1")
    assert doc == {"folder": "/u/src/app", "used": doc["used"]}
    assert isinstance(doc["used"], int)


def test_private_homes_are_listed_newest_first(home):
    import json
    for project, used in (("a-1", 100), ("b-2", 300)):
        settings.private_home("pi", project)
        settings.project_record_path("pi", project).write_text(
            json.dumps({"folder": f"/u/{project}", "used": used}))
    settings.project_dir("omp", "no-home")               # a lock, and no home yet
    homes = settings.private_homes()
    assert [(h.client, h.project, h.folder, h.used) for h in homes] == [
        ("pi", "b-2", "/u/b-2", 300), ("pi", "a-1", "/u/a-1", 100)]


@pytest.mark.parametrize("mac,theme", [({"theme": "light", "projects": {"/x": {}}}, "light"),
                                       ({"projects": {}}, None), (None, None)])
def test_a_claude_code_home_starts_with_the_onboarding_done(home, mac, theme):
    import json
    if mac is not None:
        (home / ".claude.json").write_text(json.dumps(mac))
    private = settings.private_home("claude-code", "proj-1")
    settings.ready_home("claude-code", private)
    doc = json.loads((private / ".claude.json").read_text())
    assert doc == {"hasCompletedOnboarding": True, **({"theme": theme} if theme else {})}
    (private / ".claude.json").write_text("{}")          # the client's own file stays
    settings.ready_home("claude-code", private)
    assert (private / ".claude.json").read_text() == "{}"
    settings.ready_home("pi", settings.private_home("pi"))
    assert not (settings.private_home("pi") / ".claude.json").exists()


def test_a_claude_code_home_with_a_planted_link_is_left_alone(home, tmp_path):
    private = settings.private_home("claude-code", "proj-1")
    target = tmp_path / "mac-file"
    (private / ".claude.json").symlink_to(target)
    settings.ready_home("claude-code", private)
    assert not target.exists()


# Volumes and forwarded ports

def test_volumes_get_the_default_size_and_may_nest_in_a_share(home):
    proj = os.path.realpath(home / "src" / "proj")
    cfg = LaunchClientCfg(volumes=["pg:/var/lib/postgresql", f"nm:{proj}/node_modules:8G"])
    vols = {m.source: m for m in _plan(home, cfg=cfg).volumes}
    assert vols["pg"].size == "32G" and vols["nm"].size == "8G"


@pytest.mark.parametrize("clash", ["share", "home", "volume"])
def test_volume_target_clashes_are_refused(home, clash):
    proj = os.path.realpath(home / "src" / "proj")
    home_dir = home / ".local" / "share" / "gmlx" / "launch" / "pi" / "projects" / "default" / "home"
    target = {"share": proj, "home": str(home_dir), "volume": "/v"}[clash]
    vols = [f"a:{target}"] + (["b:/v"] if clash == "volume" else [])
    with pytest.raises(SettingsError, match="both use"):
        _plan(home, cfg=LaunchClientCfg(volumes=vols))


def test_forward_refuses_the_server_and_web_ports(home):
    with pytest.raises(SettingsError, match="server's port"):
        _plan(home, cfg=LaunchClientCfg(forward=[8080]), api_port=8080)
    with pytest.raises(SettingsError, match="web app's own port"):
        _plan(home, cfg=LaunchClientCfg(forward=[3000]), web_port=3000)
    assert _plan(home, cfg=LaunchClientCfg(forward=[5432, 5432])).forward == [5432]


# Git worktrees

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_linked_worktree_shares_its_git_folder(home):
    repo = home / "src" / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=repo)
    wt = home / "src" / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=repo)
    plan = _plan(home, cwd=str(wt))
    git = [m for m in plan.mounts if m.kind == "git"]
    assert [m.source for m in git] == [os.path.realpath(repo / ".git")]
    assert not git[0].readonly
    assert git[0].note == "the git folder of this worktree of ~/src/repo"
    (wt / "sub").mkdir()
    below = _plan(home, cwd=str(wt / "sub"))
    assert not [m for m in below.mounts if m.kind == "git"]
    assert any("needs the repository root" in n for n in below.notes)
    main = _plan(home, cwd=str(repo))                  # a plain repo needs nothing
    assert not [m for m in main.mounts if m.kind == "git"] and not main.notes


def test_git_is_ignored_when_the_repo_root_is_refused(home):
    _git("init", "-q", cwd=home)                       # a dotfiles repo at $HOME
    plan = _plan(home)
    assert not [m for m in plan.mounts if m.kind == "git"]
    assert not any("git" in n for n in plan.notes)


def test_a_worktree_of_a_dotfiles_repo_never_shares_the_home_git_folder(home):
    _git("init", "-q", "-b", "main", cwd=home)         # a dotfiles repo at $HOME
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=home)
    wt = home / "src" / "dots"
    _git("worktree", "add", "-q", str(wt), cwd=home)
    plan = _plan(home, cwd=str(wt))
    assert not [m for m in plan.mounts if m.kind == "git"]
    assert any("cannot reach this repository's git folder ~/.git" in n
               and "your home folder" in n for n in plan.notes)


def test_a_submodule_shares_its_git_folder(home):
    lib = home / "src" / "lib"
    lib.mkdir()
    _git("init", "-q", "-b", "main", cwd=lib)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=lib)
    top = home / "src" / "top"
    top.mkdir()
    _git("init", "-q", "-b", "main", cwd=top)
    _git("-c", "protocol.file.allow=always", "submodule", "add", "-q", str(lib), "lib",
         cwd=top)
    plan = _plan(home, cwd=str(top / "lib"))
    git = [m for m in plan.mounts if m.kind == "git"]
    assert [m.source for m in git] == [os.path.realpath(top / ".git" / "modules" / "lib")]
    assert git[0].note == "the git folder of this submodule of ~/src/top"


def _private_repo(home):
    repo = home / "work" / "private"
    repo.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=repo)
    return repo


def _proj_git_mounts(home, proj):
    plan = _plan(home, cwd=str(proj))
    return [m for m in plan.mounts if m.kind == "git"], plan.notes


def test_a_commondir_file_in_a_shared_git_folder_mounts_nothing(home):
    """The guest writes a commondir file into the project's .git folder."""
    private = _private_repo(home)
    proj = home / "src" / "proj"
    _git("init", "-q", "-b", "main", cwd=proj)
    (proj / ".git" / "commondir").write_text(str(private / ".git") + "\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("cannot use the git folder ~/work/private/.git" in n and "--mount" in n
               for n in notes)


def test_a_gitfile_naming_another_repository_mounts_nothing(home):
    private = _private_repo(home)
    proj = home / "src" / "proj"
    (proj / ".git").write_text(f"gitdir: {private / '.git'}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("does not name ~/src/proj" in n and "--mount ~/work/private/.git" in n
               for n in notes)


def test_a_gitfile_naming_another_worktree_mounts_nothing(home):
    private = _private_repo(home)
    other = home / "work" / "other-wt"
    _git("worktree", "add", "-q", str(other), cwd=private)
    proj = home / "src" / "proj"
    wt_id = next((private / ".git" / "worktrees").iterdir()).name
    (proj / ".git").write_text(f"gitdir: {private / '.git' / 'worktrees' / wt_id}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("--mount" in n for n in notes)


def test_a_worktree_with_relative_paths_shares_its_git_folder(home):
    repo = _private_repo(home)
    wt = home / "src" / "rel-wt"
    _git("worktree", "add", "-q", "--relative-paths", str(wt), cwd=repo)
    assert not (repo / ".git" / "worktrees" / "rel-wt" / "gitdir").read_text().startswith("/")
    git, notes = _proj_git_mounts(home, wt)
    assert [m.source for m in git] == [os.path.realpath(repo / ".git")]


@pytest.mark.parametrize("link", ["folder", "dotgit"])
def test_a_link_in_the_share_never_claims_another_worktree(home, link):
    """Another repository has a worktree inside the share. The guest links
    that worktree to the project and points the project's .git at it."""
    other = _private_repo(home)
    proj = home / "src" / "proj"
    wt = proj / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=other)
    entry = other / ".git" / "worktrees" / "wt"
    if link == "folder":
        shutil.rmtree(wt)
        wt.symlink_to(proj, target_is_directory=True)
    else:
        (wt / ".git").unlink()
        (wt / ".git").symlink_to(proj / ".git")
    (proj / ".git").write_text(f"gitdir: {entry}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []


def test_a_submodule_whose_worktree_holds_no_git_file_mounts_nothing(home):
    """git takes core.worktree as the repository root, so the root is in the
    share, but no .git file there names the git folder."""
    lib = _private_repo(home)
    proj = home / "src" / "proj"
    (proj / "inner").mkdir()
    subprocess.run(["git", "config", "--file", str(lib / ".git" / "config"),
                    "core.worktree", str(proj / "inner")], check=True)
    (proj / ".git").write_text(f"gitdir: {lib / '.git'}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("does not name ~/src/proj/inner" in n for n in notes)


def _victim_worktree(home, at, *, relative=False):
    """A worktree of a repository outside every share, added at ``at``."""
    victim = _private_repo(home)
    at.parent.mkdir(parents=True, exist_ok=True)
    flags = ["--relative-paths"] if relative else []
    _git("worktree", "add", "-q", *flags, str(at), cwd=victim)
    return victim, next((victim / ".git" / "worktrees").iterdir())


@pytest.mark.parametrize("case", ["absolute", "relative", "parent"])
def test_a_link_left_by_an_earlier_launch_never_claims_another_worktree(home, case):
    """An earlier launch shared ~/area, which holds a worktree of another
    repository. The guest replaced that worktree with a link and points a
    new project's .git at the worktree entry. This launch shares only the
    new project, so the link lies outside its shares."""
    area = home / "area"
    if case == "parent":
        # The link names the parent of the folder this launch shares.
        victim, entry = _victim_worktree(home, area / "wt" / "p2")
        shutil.rmtree(area / "wt")
        (area / "x" / "p2").mkdir(parents=True)
        (area / "wt").symlink_to(area / "x", target_is_directory=True)
        proj = area / "x" / "p2"
    else:
        victim, entry = _victim_worktree(home, area / "wt", relative=case == "relative")
        shutil.rmtree(area / "wt")
        proj = area / "p2"
        proj.mkdir()
        (area / "wt").symlink_to(proj, target_is_directory=True)
    (proj / ".git").write_text(f"gitdir: {entry}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    tp = "~/" + str(proj.relative_to(home))
    assert any("cannot use the git folder ~/work/private/.git" in n
               and f"names {tp} only through a symbolic link" in n
               and "git worktree repair" in n for n in notes)


def test_a_link_in_the_private_home_never_claims_another_worktree(home):
    other = _private_repo(home)
    proj = home / "src" / "proj"
    wt = settings.private_home("pi") / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=other)
    entry = other / ".git" / "worktrees" / "wt"
    shutil.rmtree(wt)
    wt.symlink_to(proj, target_is_directory=True)
    (proj / ".git").write_text(f"gitdir: {entry}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("only through a symbolic link" in n for n in notes)


def test_a_submodule_named_through_a_link_mounts_nothing(home):
    lib = _private_repo(home)
    proj = home / "src" / "proj"
    (proj / "sub").symlink_to(proj, target_is_directory=True)
    subprocess.run(["git", "config", "--file", str(lib / ".git" / "config"),
                    "core.worktree", str(proj / "sub")], check=True)
    (proj / ".git").write_text(f"gitdir: {lib / '.git'}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("only through a symbolic link" in n and "set core.worktree in" in n
               for n in notes)


def test_a_worktree_moved_by_hand_is_shared_after_git_worktree_repair(home):
    repo = _private_repo(home)
    old = home / "src" / "old-wt"
    _git("worktree", "add", "-q", str(old), cwd=repo)
    new = home / "src" / "new-wt"
    old.rename(new)
    old.symlink_to(new, target_is_directory=True)
    git, notes = _proj_git_mounts(home, new)
    assert git == [] and any("run git worktree repair there" in n for n in notes)
    _git("worktree", "repair", cwd=new)
    git, notes = _proj_git_mounts(home, new)
    assert [m.source for m in git] == [os.path.realpath(repo / ".git")]


def test_git_runs_with_fsmonitor_off(home, monkeypatch):
    seen = []
    real = subprocess.run

    def spy(argv, *a, **k):
        seen.append(argv)
        return real(argv, *a, **k)
    monkeypatch.setattr(settings.subprocess, "run", spy)
    _plan(home)
    assert seen and all(a[:3] == ["git", "-c", "core.fsmonitor=false"] for a in seen
                        if a[0] == "git")


def test_a_mount_in_another_case_still_covers_a_worktree_git_folder(home):
    if not (home / "SRC").exists():                   # probe the volume, not the code
        pytest.skip("this volume compares names with case")
    repo = home / "src" / "R"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=repo)
    wt = home / "src" / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=repo)
    plan = _plan(home, cwd=str(wt), cli_mounts=[str(home / "SRC" / "R") + ":ro"])
    assert not [m for m in plan.mounts if m.kind == "git"]     # the ro mount covers it


# Protected folders and memory

def test_protected_folder_warning(home):
    docs = home / "Documents" / "proj"
    docs.mkdir(parents=True)
    (home / "Documents" / "other").mkdir()
    plan = _plan(home, cwd=str(docs), cli_mounts=["~/Documents/other"])
    assert plan.warnings == [                           # one line for the guarded folder
        "[launch] ~/Documents/other is in ~/Documents, which macOS guards. macOS may ask once "
        "whether the container runtime can read it, and the container waits until you answer."]
    assert plan.warnings[0].key == f"guarded:{os.path.realpath(home / 'Documents')}"
    assert not _plan(home).warnings


def test_a_share_of_the_guarded_folder_itself_reads_right(home):
    (home / "Downloads").mkdir()
    plan = _plan(home, cwd=str(home / "Downloads"))
    assert plan.warnings[0].startswith("[launch] ~/Downloads is a folder that macOS guards. ")


def test_memory_warning(monkeypatch):
    assert settings.memory_warning("1024G") is not None
    assert settings.memory_warning("1G") is None
    # A 16 GB Mac: the 4G default is exactly a quarter, so it does not warn.
    pages = {"SC_PAGE_SIZE": 16384, "SC_PHYS_PAGES": (16 << 30) // 16384}
    monkeypatch.setattr(settings.os, "sysconf", lambda name: pages[name])
    assert settings.memory_warning("4G") is None
    assert settings.memory_warning("4097M") is not None
    assert settings.memory_warning("8G").key == settings.memory_warning("8192M").key


# The guest environment and the private home

def test_guest_env_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("TZ", "Europe/Berlin")
    monkeypatch.setenv("COLORTERM", "truecolor")
    env = settings.guest_env(tmp_path)
    assert env == {"HOME": str(tmp_path), "TERM": "xterm-kitty", "LANG": "C.UTF-8",
                   "COLORTERM": "truecolor", "TZ": "Europe/Berlin"}
    monkeypatch.setenv("TERM", "bad term;x")
    assert settings.guest_env(tmp_path)["TERM"] == "xterm-256color"


def test_seed_copies_then_adds_the_git_identity(home):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = host@example.com\n")
    (home / ".claude").mkdir()
    (home / ".claude" / "CLAUDE.md").write_text("rules")
    private = settings.private_home("claude-code")
    assert settings.seed_home(private, ["~/.claude/CLAUDE.md"]) == [
        "[launch] seed: copied ~/.claude/CLAUDE.md into the private home"]
    assert (private / ".claude" / "CLAUDE.md").read_text() == "rules"
    assert settings.seed_home(private, ["~/.claude/CLAUDE.md"]) == []

    def get(key):
        return subprocess.run(["git", "config", "--file", str(private / ".gitconfig"), key],
                              capture_output=True, text=True).stdout.strip()
    assert (get("user.name"), get("user.email")) == ("Host Name", "host@example.com")


def test_seeded_gitconfig_wins_over_the_host_identity(home):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n")
    private = settings.private_home("pi")
    seeded = home / "seed"
    seeded.mkdir()
    (private / ".gitconfig").write_text("[user]\n\tname = Seeded\n")
    settings.seed_home(private, [])
    out = subprocess.run(["git", "config", "--file", str(private / ".gitconfig"), "user.name"],
                         capture_output=True, text=True).stdout.strip()
    assert out == "Seeded"


def _git_get(path, key):
    return subprocess.run(["git", "config", "--file", str(path), key],
                          capture_output=True, text=True).stdout.strip()


def test_a_git_identity_launch_wrote_follows_the_mac(home):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = old@example.com\n")
    private = settings.private_home("pi")
    assert settings.seed_home(private, []) == []
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = new@example.com\n")
    assert settings.seed_home(private, []) == [
        "[launch] updated the git user.email in the private home to match the Mac."]
    assert _git_get(private / ".gitconfig", "user.email") == "new@example.com"
    assert settings.seed_home(private, []) == []


def test_a_git_identity_set_in_the_container_stays(home):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = old@example.com\n")
    private = settings.private_home("pi")
    settings.seed_home(private, [])
    subprocess.run(["git", "config", "--file", str(private / ".gitconfig"), "user.email",
                    "work@example.com"], check=True)
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = new@example.com\n")
    assert settings.seed_home(private, []) == []
    assert _git_get(private / ".gitconfig", "user.email") == "work@example.com"


def test_a_home_from_before_the_record_follows_the_mac_once_they_match(home):
    (home / ".gitconfig").write_text("[user]\n\temail = same@example.com\n")
    private = settings.private_home("pi")
    (private / ".gitconfig").write_text("[user]\n\temail = same@example.com\n")
    assert not settings.identity_record_path(private).exists()
    assert settings.seed_home(private, []) == []
    (home / ".gitconfig").write_text("[user]\n\temail = moved@example.com\n")
    assert settings.seed_home(private, []) == [
        "[launch] updated the git user.email in the private home to match the Mac."]
    assert _git_get(private / ".gitconfig", "user.email") == "moved@example.com"


def test_seed_outside_home_or_sensitive_is_refused(home, tmp_path):
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="not inside your home"):
        settings.seed_home(private, [str(tmp_path)])
    (home / ".npmrc").write_text("//registry/:_authToken=x")
    with pytest.raises(SettingsError, match="will not copy ~/.npmrc, because it holds credentials"):
        settings.seed_home(private, ["~/.npmrc"])
    assert not (private / ".npmrc").exists()


def test_seed_never_writes_through_a_link_the_guest_planted(home, tmp_path):
    (home / ".claude").mkdir()
    (home / ".claude" / "CLAUDE.md").write_text("rules")
    private = settings.private_home("claude-code")
    outside = tmp_path / "mac-folder"
    outside.mkdir()
    (private / ".claude").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SettingsError, match="symbolic link"):
        settings.seed_home(private, ["~/.claude/CLAUDE.md"])
    assert list(outside.iterdir()) == []


def test_git_identity_never_writes_through_a_planted_gitconfig(home, tmp_path):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n")
    private = settings.private_home("pi")
    secret = tmp_path / "mac-gitconfig"
    secret.write_text("[core]\n")
    (private / ".gitconfig").symlink_to(secret)
    warns = settings.seed_home(private, [])
    assert len(warns) == 1 and "symbolic link" in warns[0] and "git identity" in warns[0]
    assert secret.read_text() == "[core]\n"
    assert not list(tmp_path.glob("mac-gitconfig.lock"))


@pytest.mark.parametrize("plant", ["latin1", "socket"])
def test_an_unreadable_gitconfig_only_warns(home, plant, tmp_path):
    import socket
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n")
    private = settings.private_home("pi")
    target = private / ".gitconfig"
    if plant == "latin1":
        target.write_bytes(b"[user]\n\tname = J\xf6rg\n")
        expect = "not UTF-8 text"
    else:
        import tempfile
        sock = socket.socket(socket.AF_UNIX)
        short = os.path.join(tempfile.mkdtemp(dir="/tmp"), "s")   # a socket path stays short
        sock.bind(short)
        os.rename(short, target)
        os.rmdir(os.path.dirname(short))
        expect = "cannot be used"
    try:
        warns = settings.seed_home(private, [])
    finally:
        if plant == "socket":
            sock.close()
    assert len(warns) == 1 and expect in warns[0]


@pytest.mark.parametrize("project", ["default", "proj-1234abcd"])
def test_confine_refuses_a_private_home_outside_confined(home, project):
    from gmlx.container import confine
    private = settings.private_home("pi", project)
    for call in (lambda: confine.read_text(private / "x"),
                 lambda: confine.write_text(private / "x", "y"),
                 lambda: confine.exists(private / "x"),
                 lambda: confine.mkdirs(private / "d"),
                 lambda: confine.listdir(private)):
        with pytest.raises(confine.ConfinedError, match="private home"):
            call()
    confine.write_text(home / "ok.txt", "fine")          # the Mac's own files still work
    assert (home / "ok.txt").read_text() == "fine"


def test_confine_ignores_case_on_a_volume_that_ignores_it(home):
    from gmlx.container import confine
    if not (home.parent / "HOME").exists():            # probe the volume, not the code
        pytest.skip("this volume compares names with case")
    private = settings.private_home("pi")
    other_data = Path(str(private).replace("/.local/share/", "/.LOCAL/share/"))
    for path in (other_data / "x", private.parent / "HOME" / "x"):
        with pytest.raises(confine.ConfinedError, match="private home"):
            confine.exists(path)
    (home / "dots").mkdir()                              # a dotfiles link, spelled in
    (home / ".rc").symlink_to(home.parent / "HOME" / "dots" / "rc")   # another case
    confine.write_text(home / ".rc", "x")
    assert (home / "dots" / "rc").read_text() == "x"


def test_seed_refuses_the_launch_data_folder(home):
    private = settings.private_home("pi")
    settings.private_home("omp")
    with pytest.raises(SettingsError, match="private homes of the clients"):
        settings.seed_home(private, ["~/.local"])
    with pytest.raises(SettingsError, match="private homes of the clients"):
        settings.seed_home(private, ["~/.local/share/gmlx/launch/omp"])


def test_a_relative_seed_is_read_from_home(home, monkeypatch):
    (home / "notes.md").write_text("n")
    monkeypatch.chdir(home / "src" / "proj")
    private = settings.private_home("pi")
    assert settings.seed_home(private, ["notes.md"]) == [
        "[launch] seed: copied ~/notes.md into the private home"]
    assert (private / "notes.md").read_text() == "n"


def test_a_failed_seed_copy_leaves_nothing_behind(home):
    tools = home / "tools"
    tools.mkdir()
    (tools / "a.txt").write_text("a")
    (tools / "z.txt").write_text("z")
    (tools / "z.txt").chmod(0)
    private = settings.private_home("pi")
    try:
        with pytest.raises(SettingsError, match="cannot copy ~/tools"):
            settings.seed_home(private, ["~/tools"])
    finally:
        (tools / "z.txt").chmod(0o600)
    assert sorted(p.name for p in private.iterdir()) == []
    settings.seed_home(private, ["~/tools"])          # the next launch copies it whole
    assert sorted(p.name for p in (private / "tools").iterdir()) == ["a.txt", "z.txt"]


def test_a_seed_copy_left_by_a_killed_launch_is_removed(home):
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")
    for n in range(2):
        left = private / f".notes.md.gmlx-seed-dead{n}"
        left.mkdir()
        (left / "part").write_text("x")
    settings.seed_home(private, ["~/notes.md"])
    assert sorted(p.name for p in private.iterdir()) == ["notes.md"]


def test_an_interrupted_seed_copy_leaves_nothing(home, monkeypatch):
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")

    def interrupted(src, dst):
        from gmlx.container import confine
        confine.write_text(dst, "half")
        raise KeyboardInterrupt
    monkeypatch.setattr(settings, "_copy_confined", interrupted)
    with pytest.raises(KeyboardInterrupt):
        settings.seed_home(private, ["~/notes.md"])
    assert list(private.iterdir()) == []


def test_a_missing_seed_is_reported_and_skipped(home):
    private = settings.private_home("pi")
    assert settings.seed_home(private, ["~/nope.md"]) == [
        "[launch] seed: ~/nope.md does not exist, so nothing was copied."]
    assert not (private / "nope.md").exists()


def test_seed_keeps_modes_and_links_and_reports_copy_errors(home):
    tools = home / "tools"
    tools.mkdir()
    (tools / "run.sh").write_text("#!/bin/sh\n")
    (tools / "run.sh").chmod(0o755)
    (tools / "latest").symlink_to("run.sh")
    os.mkfifo(tools / "pipe")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/tools"])
    assert (private / "tools" / "run.sh").stat().st_mode & 0o777 == 0o755
    assert os.readlink(private / "tools" / "latest") == "run.sh"
    assert not (private / "tools" / "pipe").exists()
    (home / "locked").write_text("x")
    (home / "locked").chmod(0)
    try:
        with pytest.raises(SettingsError, match="cannot copy ~/locked"):
            settings.seed_home(private, ["~/locked"])
    finally:
        (home / "locked").chmod(0o600)


# The server config the guest could change

def _share(path):
    return [Mount(os.path.realpath(path), os.path.realpath(path))]


def _config(path, text):
    path.write_text(text)
    return str(path)


def test_config_inside_a_share_warns(home):
    proj = home / "src" / "proj"
    cfg = _config(proj / "gmlx.yaml", "server: {port: 8080}\n")
    out = settings.server_config_warnings(cfg, _share(proj))
    assert len(out) == 1 and "in the read-write share" in out[0]
    ro = [Mount(os.path.realpath(proj), os.path.realpath(proj), readonly=True)]
    assert settings.server_config_warnings(cfg, ro) == []


def test_scan_folder_warnings(home, monkeypatch):
    import gmlx.load.discovery as discovery

    def boom(*a, **k):
        raise AssertionError("the check must not scan")
    monkeypatch.setattr(discovery, "merge_discovered", boom)
    monkeypatch.setattr(os, "walk", boom)
    monkeypatch.setattr(os, "listdir", boom)
    proj = home / "src" / "proj"
    models = home / "models"
    (models / "sub").mkdir(parents=True)
    cases = [
        (f"discover: [{{dir: {proj}/gguf}}]\n", _share(proj), True),
        (f"server: {{model_dirs: [{proj}]}}\ndiscover: [{{}}]\n", _share(proj), True),
        (f"discover: [{{dir: {models}, recursive: true}}]\n", _share(models / "sub"), True),
        (f"discover: [{{dir: {models}}}]\n", _share(models / "sub"), False),
    ]
    for text, shares, warned in cases:
        cfg = _config(home / "gmlx.yaml", text)
        out = settings.server_config_warnings(cfg, shares)
        assert any("scans" in w for w in out) is warned, text


def test_model_file_inside_a_share_warns(home):
    proj = home / "src" / "proj"
    (proj / "m.gguf").write_text("x")
    cfg = _config(home / "gmlx.yaml",
                  f"server: {{model_dirs: [{proj}]}}\nmodels:\n  m: {{path: m.gguf}}\n")
    out = settings.server_config_warnings(cfg, _share(proj))
    assert any("model file" in w for w in out)


def test_relative_config_paths_resolve_from_the_config_folder(home, monkeypatch):
    """The server runs in its config file's folder, so a relative model path
    or model folder never lands in the share launch runs from."""
    proj = home / "src" / "proj"
    monkeypatch.chdir(proj)
    (proj / "canary.gguf").write_text("x")
    (proj / "gguf").mkdir()
    conf = home / ".config" / "gmlx"
    conf.mkdir(parents=True)
    text = ("server: {model_dirs: [models]}\ndiscover: [{dir: gguf}]\n"
            "models:\n  m: {path: canary.gguf}\n")
    assert settings.server_config_warnings(_config(conf / "gmlx.yaml", text),
                                           _share(proj)) == []
    # A config in the share resolves there, even for a model file the client
    # has not written yet.
    (proj / "canary.gguf").unlink()
    out = settings.server_config_warnings(_config(proj / "gmlx.yaml", text), _share(proj))
    assert any("the model file ~/src/proj/canary.gguf" in w for w in out)
    assert any("scans ~/src/proj/gguf" in w for w in out)


def test_an_empty_pythonpath_entry_warns_with_a_writable_share(home, monkeypatch):
    proj = home / "src" / "proj"
    monkeypatch.setenv("PYTHONPATH", "/abs/lib:")
    out = settings.pythonpath_warnings(_share(proj))
    assert len(out) == 1 and "Remove the entry from PYTHONPATH" in out[0]
    ro = [Mount(os.path.realpath(proj), os.path.realpath(proj), readonly=True)]
    assert settings.pythonpath_warnings(ro) == []
    monkeypatch.setenv("PYTHONPATH", "/abs/lib")
    assert settings.pythonpath_warnings(_share(proj)) == []


def test_broken_config_never_stops_the_launch(home):
    proj = home / "src" / "proj"
    cfg = _config(proj / "gmlx.yaml", "server: [unclosed\n")
    out = settings.server_config_warnings(cfg, _share(proj))
    assert any("could not check" in w for w in out)
    assert any("in the read-write share" in w for w in out)


@pytest.mark.parametrize("earlier", [False, True])
def test_a_config_link_leading_out_of_a_share_is_never_read(home, earlier):
    """The client replaced the config in its share with a link to a file of
    yours. Launch keeps the in-share warning and reads nothing through it."""
    proj = home / "src" / "proj"
    secret = _config(home / "secret.yaml", "server: {port: SECRET-VALUE}\n")
    (proj / "gmlx.yaml").symlink_to(secret)
    if earlier:
        settings.record_shares(SimpleNamespace(mounts=_share(proj)))
    shares = [] if earlier else _share(proj)
    out = settings.server_config_warnings(str(proj / "gmlx.yaml"), shares)
    assert any("may have replaced it with a symbolic link" in w
               and "leads to ~/secret.yaml" in w for w in out)
    assert not any("SECRET" in w or "could not check" in w for w in out)
    assert any("in the read-write share" in w for w in out) is not earlier


def test_a_config_link_of_your_own_is_read(home):
    """A dotfiles link outside every share, and a link that stays in the
    share, are read as usual."""
    proj = home / "src" / "proj"
    (home / "dotfiles").mkdir()
    real = _config(home / "dotfiles" / "gmlx.yaml", "server: [unclosed\n")
    conf = home / ".config" / "gmlx"
    conf.mkdir(parents=True)
    (conf / "gmlx.yaml").symlink_to(real)
    out = settings.server_config_warnings(str(conf / "gmlx.yaml"), _share(proj))
    assert len(out) == 1 and "could not check" in out[0]      # it was parsed
    _config(proj / "base.yaml", "server: {port: 8080}\n")
    (proj / "gmlx.yaml").symlink_to(proj / "base.yaml")
    out = settings.server_config_warnings(str(proj / "gmlx.yaml"), _share(proj))
    assert len(out) == 1 and "in the read-write share" in out[0]


def test_fifo_and_large_config_give_the_could_not_check_line(home):
    fifo = home / "fifo.yaml"
    os.mkfifo(fifo)
    result = {}
    t = threading.Thread(target=lambda: result.update(
        out=settings.server_config_warnings(str(fifo), [])))
    t.start()
    t.join(5)
    assert not t.is_alive(), "the check blocked on a FIFO"
    assert "could not check" in result["out"][0]
    big = _config(home / "big.yaml", "#" * (settings.CONFIG_READ_MAX + 1))
    assert "could not check" in settings.server_config_warnings(big, [])[0]


def test_server_config_path_prefers_the_running_server(home, monkeypatch):
    from gmlx.serve import lifecycle
    running = _config(home / "running.yaml", "server: {}\n")
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: {"config_abspath": running, "pid": os.getpid()})
    assert settings.server_config_path("127.0.0.1", 8080) == running
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    (home / ".config" / "gmlx").mkdir(parents=True)
    user = _config(home / ".config" / "gmlx" / "gmlx.yaml", "server: {}\n")
    monkeypatch.chdir(home / "src" / "proj")
    assert settings.server_config_path("127.0.0.1", 8080) == user
    # --base-url never autostarts, so only a runfile names its config.
    assert settings.server_config_path("box.local", 8000, autostart=False) is None


def test_a_running_server_without_a_config_file_is_reported(home, monkeypatch):
    """Such a server may scan its --models-dir, which may be shared, so
    launch says it cannot check it rather than checking the wrong file. The
    note never steers to --config."""
    from gmlx.serve import lifecycle
    (home / ".config" / "gmlx").mkdir(parents=True)
    _config(home / ".config" / "gmlx" / "gmlx.yaml", "server: {}\n")
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: {"config_abspath": None, "pid": os.getpid()})
    notes: list[str] = []
    assert settings.server_config_path("127.0.0.1", 8080, notes=notes) is None
    assert any("has no config file" in n and n.endswith("Start the server from a config "
                                                         "file to have it checked.")
               for n in notes)
    assert not any("--config" in n for n in notes)


# Round-eight review: seeds, links from earlier launches, firmlinks

def _alias(path) -> str:
    """The firmlink form of ``path`` that ``os.path.realpath`` keeps."""
    alias = "/System/Volumes/Data" + os.path.realpath(path)
    if not os.path.isdir(os.path.dirname(alias)):
        pytest.skip("no /System/Volumes/Data firmlink for this folder")
    return alias


def test_a_seeded_link_turned_to_a_credential_folder_is_refused(home):
    """The user seeds ~/.config/nvim, a link into ~/dotfiles. A client in
    an earlier launch of ~/dotfiles replaced ~/dotfiles/nvim with a link to
    ~/.ssh and deleted its own copy."""
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY")
    (home / "dotfiles" / "nvim").mkdir(parents=True)
    (home / ".config").mkdir()
    (home / ".config" / "nvim").symlink_to(home / "dotfiles" / "nvim")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/.config/nvim"])
    shutil.rmtree(home / "dotfiles" / "nvim")
    (home / "dotfiles" / "nvim").symlink_to(home / ".ssh")
    shutil.rmtree(private / ".config" / "nvim")
    # The seed was copied once, so a later launch skips it without a check.
    assert settings.seed_home(private, ["~/.config/nvim"]) == []
    with pytest.raises(SettingsError, match="which holds credentials"):
        settings.seed_home(private, ["~/.config/nvim"], reseed=True)
    assert not (private / ".config" / "nvim").exists()


def test_a_seed_through_a_link_out_of_home_is_refused(home, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret").write_text("s")
    (home / "notes").symlink_to(outside)
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="outside your home folder"):
        settings.seed_home(private, ["~/notes"])
    assert not (private / "notes").exists()


def test_a_seed_changed_on_the_mac_is_copied_again(home):
    (home / ".claude").mkdir()
    (home / ".claude" / "CLAUDE.md").write_text("rules")
    private = settings.private_home("claude-code")
    copy = private / ".claude" / "CLAUDE.md"
    settings.seed_home(private, ["~/.claude/CLAUDE.md"])
    (home / ".claude" / "CLAUDE.md").write_text("rules v2")
    assert settings.seed_home(private, ["~/.claude/CLAUDE.md"]) == [
        "[launch] seed: copied ~/.claude/CLAUDE.md again, because it changed on the Mac"]
    assert copy.read_text() == "rules v2"
    assert settings.seed_home(private, ["~/.claude/CLAUDE.md"]) == []


def test_a_seed_changed_on_both_sides_keeps_the_copy_and_names_reseed(home):
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/notes.md"])
    (home / "notes.md").write_text("mac edit")
    (private / "notes.md").write_text("client edit")
    assert settings.seed_home(private, ["~/notes.md"]) == [
        "[launch] seed: ~/notes.md changed on the Mac and in the private home, so launch kept "
        "the copy in the private home. --reseed replaces it with the Mac file."]
    assert settings.seed_home(private, ["~/notes.md"]) == []           # once for each change
    assert (private / "notes.md").read_text() == "client edit"
    assert settings.seed_home(private, ["~/notes.md"], reseed=True) == [
        "[launch] seed: copied ~/notes.md again for --reseed"]
    assert (private / "notes.md").read_text() == "mac edit"


def test_a_changed_folder_seed_is_copied_again(home):
    deep = home / "tools" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "x.txt").write_text("1")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/tools"])
    (deep / "x.txt").write_text("22")                # deep inside, the top folder unchanged
    assert settings.seed_home(private, ["~/tools"]) == [
        "[launch] seed: copied ~/tools again, because it changed on the Mac"]
    assert (private / "tools" / "a" / "b" / "x.txt").read_text() == "22"


def test_a_copy_recorded_before_the_stamps_refreshes_from_then_on(home):
    import json
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")
    (private / "notes.md").write_text("old copy")
    settings.seed_record_path(private).write_text(
        json.dumps({"seeded": [str(home / "notes.md")]}))
    assert settings.seed_home(private, ["~/notes.md"]) == []
    assert (private / "notes.md").read_text() == "old copy"
    (home / "notes.md").write_text("new")
    settings.seed_home(private, ["~/notes.md"])
    assert (private / "notes.md").read_text() == "new"


def test_a_seed_a_client_replaced_with_a_link_keeps_the_copy(home):
    proj = os.path.realpath(home / "src" / "proj")
    (home / ".zsh_history").write_text("secret\n")
    (home / "src" / "proj" / "tool.conf").write_text("ok\n")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/src/proj/tool.conf"], writable=[proj])
    (home / "src" / "proj" / "tool.conf").unlink()
    (home / "src" / "proj" / "tool.conf").symlink_to(home / ".zsh_history")
    out = settings.seed_home(private, ["~/src/proj/tool.conf"], writable=[proj])
    assert len(out) == 1 and out[0].startswith("[launch] seed: will not copy ~/src/proj/tool")
    assert out[0].endswith(" Launch kept the earlier copy.")
    assert (private / "src" / "proj" / "tool.conf").read_text() == "ok\n"


def test_a_seed_the_client_deleted_is_copied_again_only_on_request(home):
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/notes.md"])
    record = settings.seed_record_path(private)
    assert not settings._inside(str(record), str(private))
    (private / "notes.md").unlink()
    (home / "notes.md").write_text("changed")
    settings.seed_home(private, ["~/notes.md"])
    assert not (private / "notes.md").exists()
    settings.seed_home(private, ["~/notes.md"], reseed=True)
    assert (private / "notes.md").read_text() == "changed"
    (private / "notes.md").write_text("client edit")
    settings.seed_home(private, ["~/notes.md"], reseed=True)       # replaces the copy
    assert (private / "notes.md").read_text() == "changed"


def test_a_sparse_seed_is_refused_before_it_is_read(home, monkeypatch):
    big = home / "big.bin"
    with open(big, "wb") as f:
        f.truncate(1 << 30)
    read = []
    real_read = os.read
    monkeypatch.setattr(settings.os, "read", lambda fd, n: read.append(n) or real_read(fd, n))
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="larger than 64 MiB"):
        settings.seed_home(private, ["~/big.bin"])
    assert read == [] and list(private.iterdir()) == []


def test_a_seed_with_too_many_files_is_refused(home, monkeypatch):
    monkeypatch.setattr(settings, "SEED_MAX_FILES", 5)
    tools = home / "tools"
    tools.mkdir()
    for n in range(10):
        (tools / f"f{n}").write_text("x")
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="more than 5 files"):
        settings.seed_home(private, ["~/tools"])
    assert list(private.iterdir()) == []


def test_a_seed_copy_streams_the_file(home, monkeypatch):
    (home / "data.bin").write_bytes(os.urandom(3 << 20))
    sizes = []
    real_read = os.read
    monkeypatch.setattr(settings.os, "read",
                        lambda fd, n: sizes.append(n) or real_read(fd, n))
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/data.bin"])
    assert (private / "data.bin").read_bytes() == (home / "data.bin").read_bytes()
    assert max(sizes) <= 1 << 20


def test_a_seed_swapped_after_the_check_is_refused(home, monkeypatch):
    """The path macOS gives the open file is checked again, so a link
    swapped in between the check and the open cannot redirect the copy."""
    (home / ".ssh").mkdir()
    (home / "notes.md").write_text("n")
    monkeypatch.setattr(settings, "fd_path", lambda fd: str(home / ".ssh" / "id"))
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="which holds credentials"):
        settings.seed_home(private, ["~/notes.md"])
    assert not (private / "notes.md").exists()


def test_seeding_a_token_file_warns(home):
    (home / ".claude.json").write_text("{}")
    private = settings.private_home("claude-code")
    warns = settings.seed_home(private, ["~/.claude.json"])
    assert "[launch] warning: seed ~/.claude.json holds a sign-in token. The client can " \
           "read it." in warns
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text("{}")
    warns = settings.seed_home(private, ["~/.claude"])
    assert "[launch] warning: seed ~/.claude copies ~/.claude/.credentials.json, which " \
           "holds a sign-in token. The client can read it." in warns


def test_a_deep_seed_leftover_is_removed(home):
    (home / "notes.md").write_text("n")
    private = settings.private_home("pi")
    fd = os.open(private, os.O_RDONLY | os.O_DIRECTORY)
    os.mkdir(".notes.md.gmlx-seed-dead", dir_fd=fd)
    for _ in range(1200):                      # deeper than the recursion limit
        nxt = os.open(".notes.md.gmlx-seed-dead" if _ == 0 else "d",
                      os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        os.close(fd)
        fd = nxt
        os.mkdir("d", dir_fd=fd)
    os.close(fd)
    settings.seed_home(private, ["~/notes.md"])
    assert sorted(p.name for p in private.iterdir()) == ["notes.md"]


@pytest.mark.parametrize("mode", [0o002, 0o000])
def test_the_launch_roots_are_private_at_any_umask(home, mode):
    from gmlx.container import state
    old = os.umask(mode)
    try:
        data, cache = state.data_dir(), state.cache_dir()
    finally:
        os.umask(old)
    for root in (data, cache):
        assert root.stat().st_mode & 0o777 == 0o700
    data.chmod(0o775)                                 # an older, looser folder
    assert state.data_dir().stat().st_mode & 0o777 == 0o700


def test_locks_and_records_never_follow_a_link(home, tmp_path):
    from gmlx.container import state
    victim = tmp_path / "victim"
    victim.write_text("keep")
    lock = state.data_dir() / "x.lock"
    lock.symlink_to(victim)
    with pytest.raises(OSError):
        state.FileLock(lock)
    record = state.data_dir() / "r.json"
    state.write_record(record, b"{}")
    assert record.read_bytes() == b"{}" and victim.read_text() == "keep"
    assert record.stat().st_mode & 0o777 == 0o600


def test_new_sensitive_folders_are_refused_as_shares(home):
    for rel in ("bin/tools", ".cargo/registry", ".cache/huggingface/hub", ".codex",
                "Library/LaunchAgents", ".config/git", ".local/bin",
                "Library/Application Support/Code"):
        (home / rel).mkdir(parents=True, exist_ok=True)
        with pytest.raises(SettingsError, match="holds (credentials|files the Mac runs)"):
            _plan(home, cwd=str(home / rel))
    assert {"/opt/homebrew", "/usr/local"} <= set(settings.sensitive_paths(str(home)))
    # A folder that holds both kinds names both, and only those.
    (home / "Library" / "Keychains").mkdir(parents=True)
    with pytest.raises(SettingsError, match=r"which hold credentials and files the Mac "
                                            r"runs\. Launch from"):
        _plan(home, cwd=str(home / "Library"))


def test_the_firmlink_form_of_home_is_refused_as_the_current_folder(home):
    with pytest.raises(SettingsError, match="your home folder"):
        _plan(home, cwd=_alias(home))


def test_a_share_through_the_firmlink_uses_the_real_path(home):
    plan = _plan(home, cwd=_alias(home / "src" / "proj"))
    share = next(m for m in plan.mounts if m.note == "working folder")
    assert share.source == os.path.realpath(home / "src" / "proj")
    assert not share.source.startswith("/System/Volumes/Data")


def test_a_gitfile_through_the_firmlink_never_mounts_a_home_git_folder(home):
    """A dotfiles repository at $HOME and its worktree. The guest rewrites
    the worktree's .git file to name the same entry through the firmlink."""
    _git("init", "-q", "-b", "main", cwd=home)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=home)
    wt = home / "src" / "dots"
    _git("worktree", "add", "-q", str(wt), cwd=home)
    entry = next((home / ".git" / "worktrees").iterdir())
    (wt / ".git").write_text(f"gitdir: {_alias(entry)}\n")
    plan = _plan(home, cwd=str(wt))
    assert not [m for m in plan.mounts if m.kind == "git"]
    assert any("your home folder" in n for n in plan.notes)


@pytest.mark.parametrize("to", ["home", "ssh-alias", "data", "data-alias"])
def test_an_explicit_mount_replaced_by_a_link_is_refused(home, to):
    """A config mount of ~/work/data that a client of an earlier launch of
    ~/work replaced with a link."""
    (home / ".ssh").mkdir()
    target = {"home": str(home), "ssh-alias": None, "data": str(settings.data_dir()),
              "data-alias": None}[to]
    if to == "ssh-alias":
        target = _alias(home / ".ssh")
    if to == "data-alias":
        target = _alias(settings.data_dir())
    (home / "work").mkdir()
    (home / "work" / "data").symlink_to(target)
    cfg = LaunchClientCfg(mounts=[f"{home}/work/data:/data:ro"])
    with pytest.raises(SettingsError, match="symbolic link"):
        _plan(home, cfg=cfg)


def test_an_explicit_mount_of_the_launch_data_folder_is_refused(home):
    settings.private_home("pi")
    data = settings.data_dir()
    for spec in (str(data), f"{data}/pi:/x", f"{home}/.local/share:/y", "~"):
        with pytest.raises(SettingsError, match="private homes of the clients") as e:
            _plan(home, cli_mounts=[spec])
        assert str(e.value).endswith(". Share a project folder instead.")


@pytest.mark.parametrize("rel", [".cache", ".cache/gmlx", ".cache/gmlx/media", ".config",
                                 ".config/gmlx", "xdg-cache", "xdg-cache/gmlx"])
def test_an_explicit_mount_of_gmlx_settings_or_server_state_is_refused(home, monkeypatch, rel):
    """The runfile there names the config file that later launches read."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "xdg-cache"))
    for folder in (".cache/gmlx/media", ".config/gmlx", "xdg-cache/gmlx"):
        (home / folder).mkdir(parents=True, exist_ok=True)
    with pytest.raises(SettingsError, match="settings and server state"):
        _plan(home, cli_mounts=[f"~/{rel}:ro"])
    (home / ".config" / "nvim").mkdir()
    assert _plan(home, cli_mounts=["~/.config/nvim"]).warnings == []


def test_an_explicit_mount_written_through_the_firmlink_is_refused(home):
    (home / "data").mkdir()
    with pytest.raises(SettingsError, match="real path"):
        _plan(home, cli_mounts=[_alias(home / "data") + ":/data"])


def test_a_git_folder_a_guest_made_of_a_shared_folder_is_refused(home):
    """An earlier launch shared ~/area. Its guest made ~/area itself a git
    folder whose core.worktree names ~/area/p2 (submodule path), and made
    the user folder ~/area/docs a git folder with a worktree entry for
    ~/area/p3 (worktree path). These launches share only p2 or p3."""
    area = home / "area"
    (area / "p2").mkdir(parents=True)
    (area / "HEAD").write_text("ref: refs/heads/main\n")
    (area / "objects").mkdir()
    (area / "refs").mkdir()
    (area / "config").write_text(f"[core]\n\trepositoryformatversion = 0\n"
                                 f"\tbare = false\n\tworktree = {area / 'p2'}\n")
    (area / "p2" / ".git").write_text(f"gitdir: {area}\n")
    git, notes = _proj_git_mounts(home, area / "p2")
    assert git == [] and any("not named like one" in n for n in notes)

    docs = area / "docs"
    p3 = area / "p3"
    p3.mkdir()
    for d in ("objects", "refs/heads", "worktrees/p3"):
        (docs / d).mkdir(parents=True, exist_ok=True)
    (docs / "notes.txt").write_text("the user's notes")
    (docs / "HEAD").write_text("ref: refs/heads/main\n")
    (docs / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = true\n")
    (docs / "worktrees/p3/HEAD").write_text("ref: refs/heads/main\n")
    (docs / "worktrees/p3/commondir").write_text("../..\n")
    (docs / "worktrees/p3/gitdir").write_text(str(p3 / ".git") + "\n")
    (p3 / ".git").write_text(f"gitdir: {docs / 'worktrees' / 'p3'}\n")
    git, notes = _proj_git_mounts(home, p3)
    assert git == [] and any("not named like one" in n for n in notes)


def test_a_worktree_of_a_bare_repository_shares_its_git_folder(home):
    bare = home / "repos" / "lib.git"
    bare.parent.mkdir()
    _git("init", "-q", "--bare", "-b", "main", str(bare), cwd=home)
    seed = home / "seedrepo"
    seed.mkdir()
    _git("init", "-q", "-b", "main", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=seed)
    _git("push", "-q", str(bare), "main", cwd=seed)
    wt = home / "src" / "lib-wt"
    _git("worktree", "add", "-q", str(wt), "main", cwd=bare)
    git, notes = _proj_git_mounts(home, wt)
    assert [m.source for m in git] == [os.path.realpath(bare)]


def test_a_read_only_share_keeps_the_worktree_git_folder_read_only(home):
    repo = home / "src" / "R"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=repo)
    wt = home / "src" / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=repo)
    plan = _plan(home, cwd=str(wt), cli_mounts=[f"{wt}:ro"])
    git = [m for m in plan.mounts if m.kind == "git"]
    assert len(git) == 1 and git[0].readonly
    rw = _plan(home, cwd=str(wt))
    assert not [m for m in rw.mounts if m.kind == "git"][0].readonly


def test_recheck_refuses_a_share_swapped_for_a_link(home, tmp_path):
    plan = _plan(home)
    settings.recheck_sources(plan)
    proj = home / "src" / "proj"
    proj.rename(home / "src" / "proj-moved")
    proj.symlink_to(home, target_is_directory=True)
    with pytest.raises(SettingsError, match="changed after launch checked it"):
        settings.recheck_sources(plan)


def test_the_data_volume_itself_is_a_system_folder(home):
    assert "system folder" in settings.auto_share_refusal("/System/Volumes/Data", str(home))


def test_a_seed_nested_too_deep_is_refused(home, monkeypatch):
    monkeypatch.setattr(settings, "SEED_MAX_DEPTH", 2)
    (home / "tools" / "a" / "b" / "c").mkdir(parents=True)
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="nested more than 2 folders deep"):
        settings.seed_home(private, ["~/tools"])
    assert list(private.iterdir()) == []


def test_new_folders_above_a_launch_root_are_private(home):
    from gmlx.container import state
    old = os.umask(0o002)
    try:
        data = state.data_dir()
    finally:
        os.umask(old)
    for folder in (data.parent, data.parent.parent):          # gmlx, share
        assert folder.stat().st_mode & 0o777 == 0o700


def test_confine_refuses_a_private_home_through_the_firmlink(home):
    from gmlx.container import confine
    private = settings.private_home("pi")
    (private / "x").write_text("x")
    with pytest.raises(confine.ConfinedError, match="private home"):
        confine.exists(Path(_alias(private)) / "x")


def test_confine_refuses_a_private_home_when_the_data_folder_is_named_by_the_firmlink(
        home, monkeypatch):
    from gmlx.container import confine
    (home / ".local" / "share").mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", _alias(home / ".local" / "share"))
    from gmlx.container.state import canonical
    private = Path(canonical(settings.private_home("pi")))
    assert not str(private).startswith("/System/Volumes/Data")
    (private / "x").write_text("x")
    with pytest.raises(confine.ConfinedError, match="private home"):
        confine.exists(private / "x")


# Round nine: seeds in folders a client can write, build folders, .bare

def test_a_seed_in_a_share_swapped_for_a_link_is_refused_on_reseed(home):
    """The seed lies in the shared project. The client replaces it with a
    link to the shell history, and the user runs --reseed."""
    proj = os.path.realpath(home / "src" / "proj")
    (home / ".zsh_history").write_text(": 1700000000:0;export GITHUB_TOKEN=ghp_MAC\n")
    (home / "src" / "proj" / "tool.conf").write_text("ok\n")
    private = settings.private_home("pi")
    settings.seed_home(private, ["~/src/proj/tool.conf"], writable=[proj])
    (home / "src" / "proj" / "tool.conf").unlink()
    (home / "src" / "proj" / "tool.conf").symlink_to(home / ".zsh_history")
    with pytest.raises(SettingsError) as e:
        settings.seed_home(private, ["~/src/proj/tool.conf"], reseed=True, writable=[proj])
    assert "~/src/proj/tool.conf" in str(e.value) and "~/.zsh_history" in str(e.value)
    assert (private / "src" / "proj" / "tool.conf").read_text() == "ok\n"


def test_a_folder_an_earlier_session_shared_counts_for_a_new_seed(home):
    """This launch shares nothing, but an earlier one shared the project,
    and its client left a link there."""
    (home / ".zsh_history").write_text("secret\n")
    (home / "src" / "proj" / "tool.conf").symlink_to(home / ".zsh_history")
    settings.record_shares(_plan(home))
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="which a session shared read-write"):
        settings.seed_home(private, ["~/src/proj/tool.conf"])
    assert not (private / "src" / "proj" / "tool.conf").exists()


def test_seed_writable_holds_the_shares_and_the_current_folder(home):
    plan = _plan(home, mount_cwd=False)
    proj = os.path.realpath(home / "src" / "proj")
    assert proj in settings.seed_writable(plan, str(home / "src" / "proj"))
    assert settings.seed_writable(plan, str(home)) == []        # home is never shared


def test_a_seed_through_your_own_link_names_its_real_path(home):
    (home / "dotfiles" / "nvim").mkdir(parents=True)
    (home / "dotfiles" / "nvim" / "init.lua").write_text("x")
    (home / ".config").mkdir()
    (home / ".config" / "nvim").symlink_to(home / "dotfiles" / "nvim")
    private = settings.private_home("pi")
    out = settings.seed_home(private, ["~/.config/nvim"])
    assert any("copying ~/.config/nvim from ~/dotfiles/nvim" in line for line in out)
    assert (private / ".config" / "nvim" / "init.lua").read_text() == "x"


def test_seeding_gitconfig_warns_about_tokens(home):
    (home / ".gitconfig").write_text('[url "https://tok@github.com/"]\n\tinsteadOf = gh:\n')
    private = settings.private_home("pi")
    out = settings.seed_home(private, ["~/.gitconfig"])
    assert any("~/.gitconfig" in line and "insteadOf" in line for line in out)


def test_a_share_of_a_build_folder_is_refused(home):
    box = home / "src" / "proj" / "box"
    box.mkdir()
    (box / "Containerfile").write_text("FROM x\n")
    with pytest.raises(SettingsError, match="the omp build: folder"):
        _plan(home, build_folders={"omp": str(box)})
    with pytest.raises(SettingsError, match="the omp build: folder"):
        _plan(home, build_folders={"omp": str(box / "Containerfile")})
    with pytest.raises(SettingsError, match="the omp build: folder"):   # a share inside it
        _plan(home, build_folders={"omp": str(home / "src")})
    proj = str(home / "src" / "proj")
    ro = _plan(home, build_folders={"omp": str(box)}, cli_mounts=[proj + ":ro"])
    assert all(m.readonly for m in ro.mounts if m.kind == "share")
    assert _plan(home, build_folders={"omp": str(home / "containers")}).mounts


def test_a_bare_layout_with_worktrees_shares_its_git_folder(home):
    src = _private_repo(home)
    top = home / "src" / "bare-top"
    top.mkdir()
    _git("clone", "-q", "--bare", str(src), str(top / ".bare"), cwd=home)
    (top / ".git").write_text("gitdir: ./.bare\n")
    _git("worktree", "add", "-q", str(top / "main"), cwd=top)
    git, notes = _proj_git_mounts(home, top / "main")
    assert [m.source for m in git] == [os.path.realpath(top / ".bare")]
    assert git[0].note == "the git folder of this worktree of ~/src/bare-top"


def test_a_relative_runfile_config_counts_as_unknown(home, monkeypatch):
    from gmlx.serve import lifecycle
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: {"config_abspath": "gmlx.yaml", "pid": os.getpid()})
    monkeypatch.setattr(lifecycle, "pid_alive", lambda pid: True)
    notes: list[str] = []
    assert settings.server_config_path("127.0.0.1", 8080, notes=notes) is None
    assert notes and "gmlx restart" in notes[0]
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: {"config_abspath": "/abs/gmlx.yaml", "pid": os.getpid()})
    assert settings.server_config_path("127.0.0.1", 8080, notes=[]) == "/abs/gmlx.yaml"


def test_the_gmlx_folder_above_the_roots_is_kept_private(home, monkeypatch):
    from gmlx.container import state
    gmlx_cache = home / ".cache" / "gmlx"
    gmlx_cache.mkdir(parents=True)
    os.chmod(gmlx_cache, 0o777)              # made by another command under umask 000
    state.cache_dir()
    assert os.stat(gmlx_cache).st_mode & 0o777 == 0o700


def _bare_with_worktree(home, top, worktree):
    """A bare repository at ``top/.bare`` with ``worktree`` added by git."""
    src = _private_repo(home)
    top.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(src), str(top / ".bare"), cwd=home)
    (top / ".git").write_text("gitdir: ./.bare\n")
    _git("worktree", "add", "-q", str(worktree), cwd=top)


def _shared_before(*folders, worktrees=()):
    settings.record_shares(SimpleNamespace(mounts=[
        *(Mount(os.path.realpath(f), os.path.realpath(f)) for f in folders),
        *(Mount(os.path.realpath(g), os.path.realpath(g), kind="git",
                worktree=os.path.realpath(t)) for t, g in worktrees)]))


def test_a_worktree_entry_an_earlier_share_could_forge_is_not_shared(home):
    """An earlier launch shared ~/repos read-write, so its client could have
    written both the project's .git file and the entry in the other
    repository that names the project back."""
    top, proj = home / "repos" / "other", home / "repos" / "proj" / "main"
    _bare_with_worktree(home, top, proj)
    git, _ = _proj_git_mounts(home, proj)
    assert [m.source for m in git] == [os.path.realpath(top / ".bare")]
    _shared_before(home / "repos")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []
    assert any("lies in ~/repos, which an earlier launch shared read-write" in n
               and "git worktree list in ~/repos/other" in n
               and "--mount ~/repos/other/.bare" in n for n in notes)


def test_a_worktree_beside_a_shared_main_checkout_keeps_its_git_folder(home):
    """The main checkout was shared first, then the worktree. The worktree's
    .git file named that git folder when launch first shared it."""
    repo, wt = home / "src" / "repo", home / "src" / "wt"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "--allow-empty",
         "-m", "x", cwd=repo)
    _git("worktree", "add", "-q", str(wt), cwd=repo)
    settings.record_shares(_plan(home, cwd=str(repo)))
    first = _plan(home, cwd=str(wt))
    assert [m.source for m in first.mounts if m.kind == "git"] == [
        os.path.realpath(repo / ".git")]
    settings.record_shares(first)
    assert (os.path.realpath(wt), os.path.realpath(repo / ".git")) in settings.worktree_history()
    again = _plan(home, cwd=str(wt))
    assert [m.source for m in again.mounts if m.kind == "git"] == [
        os.path.realpath(repo / ".git")]
    # A client of that session points the .git file at another repository
    # it could write, and forges the entry there.
    other = home / "src" / "other"
    _bare_with_worktree(home, other, home / "src" / "other-wt")
    settings.record_shares(_plan(home, cwd=str(other)))
    entry = other / ".bare" / "worktrees" / "other-wt"
    (entry / "gitdir").write_text(f"{os.path.realpath(wt)}/.git\n")
    (wt / ".git").write_text(f"gitdir: {os.path.realpath(entry)}\n")
    git, notes = _proj_git_mounts(home, wt)
    assert git == []
    assert any("an earlier launch shared read-write" in n for n in notes)


# ssh_agent: a socket path and the agent check

def _agent_socket(folder: Path) -> tuple:
    """A listening Unix socket in ``folder``, kept open by the caller."""
    import socket
    path = folder / "agent.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    return s, path


def test_ssh_agent_names_a_socket_of_this_user(monkeypatch):
    short = Path(tempfile.mkdtemp(prefix="ga-", dir="/tmp"))
    try:
        monkeypatch.setenv("HOME", str(short))
        sock, path = _agent_socket(short)
        with sock:
            assert settings.agent_socket("~/agent.sock", str(short)) == str(path)
            assert settings.agent_socket(True, str(short)) is None
            monkeypatch.setattr(settings.os, "getuid", lambda: os.geteuid() + 1)
            with pytest.raises(SettingsError, match=r"^ssh_agent names ~/agent.sock, which "
                                                    r"another user owns\."):
                settings.agent_socket(str(path), str(short))
        (short / "file").write_text("")
        with pytest.raises(SettingsError, match=r"^ssh_agent names ~/file, which is not a "
                                                r"socket\."):
            settings.agent_socket("~/file", str(short))
        for missing in ("~/gone.sock", "~nosuchuser/agent.sock"):
            with pytest.raises(SettingsError, match=r"which does not exist\. Start that agent"):
                settings.agent_socket(missing, str(short))
    finally:
        shutil.rmtree(short, ignore_errors=True)


def test_a_plan_keeps_the_agent_socket(home):
    short = Path(tempfile.mkdtemp(prefix="ga-", dir="/tmp"))
    try:
        sock, path = _agent_socket(short)
        with sock:
            plan = _plan(home, cfg=LaunchClientCfg(ssh_agent=str(path)))
        assert plan.ssh_agent is True and plan.ssh_socket == str(path)
        plan = _plan(home, cfg=LaunchClientCfg(ssh_agent=True))
        assert plan.ssh_agent is True and plan.ssh_socket is None
    finally:
        shutil.rmtree(short, ignore_errors=True)


@pytest.mark.parametrize("socket_path, code, line", [
    (None, 0, None),
    (None, None, None),
    (None, 1, "[launch] ssh_agent is on, but the SSH agent holds no keys, so ssh in the "
              "container cannot sign. Load one with ssh-add --apple-use-keychain "
              "~/.ssh/id_ed25519."),
    ("/tmp/own.sock", 1, "[launch] ssh_agent is on, but the SSH agent at /tmp/own.sock "
                         "holds no keys, so ssh in the container cannot sign. Load a key "
                         "into that agent."),
    (None, 2, "[launch] ssh_agent is on, but no SSH agent answers at /tmp/env.sock, so ssh "
              "in the container cannot sign."),
])
def test_the_agent_check_names_an_empty_or_silent_agent(home, monkeypatch, socket_path,
                                                         code, line):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/env.sock")
    asked = []
    monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: asked.append(sock) or code)
    plan = replace(_plan(home), ssh_agent=True, ssh_socket=socket_path)
    assert settings.agent_key_line(plan) == line
    assert asked == [socket_path or "/tmp/env.sock"]       # the socket that is forwarded


def test_the_agent_check_without_ssh_auth_sock_or_with_ssh_agent_off(home, monkeypatch):
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(settings, "_ssh_add_list", lambda sock: pytest.fail("ssh-add ran"))
    assert settings.agent_key_line(_plan(home)) is None
    plan = replace(_plan(home), ssh_agent=True)
    assert settings.agent_key_line(plan) == ("[launch] ssh_agent is on, but SSH_AUTH_SOCK is "
                                             "not set, so the container gets no SSH agent.")
