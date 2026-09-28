"""gmlx/container/settings.py: what a session shares, the refusals and
warnings, the private home, and the server-config checks."""

from __future__ import annotations

import os
import subprocess
import threading

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
    with pytest.raises(SettingsError, match="','"):
        _plan(home, cli_mounts=["~/a,b"])


def test_explicit_sensitive_mount_is_honored_with_a_warning(home):
    (home / ".ssh").mkdir()
    plan = _plan(home, cli_mounts=["~/.ssh:ro"])
    assert any(m.source.endswith("/.ssh") and m.readonly for m in plan.mounts)
    assert any("~/.ssh" in w for w in plan.warnings)


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
    with pytest.raises(SettingsError, match="both mount at /data"):
        settings.normalize_mounts([Mount("/h/a", "/data/"), Mount("/h/b", "/data")])


def test_a_read_only_mount_of_the_current_folder_replaces_the_default_share(home):
    proj = os.path.realpath(home / "src" / "proj")
    plan = _plan(home, cli_mounts=[f"{proj}:ro"])
    shares = [m for m in plan.mounts if m.kind == "share"]
    assert [(m.source, m.target, m.readonly) for m in shares] == [(proj, proj, True)]
    assert plan.workdir == proj and plan.cwd_shared


@pytest.mark.parametrize("target", ["/", "/proc", "/sys/x", "/dev", "/opt/gmlx",
                                    "/var/host-services/x", "/opt", "/var"])
def test_reserved_targets_are_refused(target):
    with pytest.raises(SettingsError):
        settings.normalize_mounts([Mount("/h/a", target)])


def test_two_mounts_at_one_target_are_refused(home):
    (home / "other").mkdir()
    proj = os.path.realpath(home / "src" / "proj")
    with pytest.raises(SettingsError, match="both mount at"):
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
    assert plan.home == home / ".local" / "share" / "gmlx" / "launch" / "elia" / "home"
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


# Volumes and forwarded ports

def test_volumes_get_the_default_size_and_may_nest_in_a_share(home):
    proj = os.path.realpath(home / "src" / "proj")
    cfg = LaunchClientCfg(volumes=["pg:/var/lib/postgresql", f"nm:{proj}/node_modules:8G"])
    vols = {m.source: m for m in _plan(home, cfg=cfg).volumes}
    assert vols["pg"].size == "32G" and vols["nm"].size == "8G"


@pytest.mark.parametrize("clash", ["share", "home", "volume"])
def test_volume_target_clashes_are_refused(home, clash):
    proj = os.path.realpath(home / "src" / "proj")
    home_dir = home / ".local" / "share" / "gmlx" / "launch" / "pi" / "home"
    target = {"share": proj, "home": str(home_dir), "volume": "/v"}[clash]
    vols = [f"a:{target}"] + (["b:/v"] if clash == "volume" else [])
    with pytest.raises(SettingsError, match="both mount at"):
        _plan(home, cfg=LaunchClientCfg(volumes=vols))


def test_forward_refuses_the_server_and_web_ports(home):
    with pytest.raises(SettingsError, match="server's port"):
        _plan(home, cfg=LaunchClientCfg(forward=[8080]), api_port=8080)
    with pytest.raises(SettingsError, match="web port"):
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
    assert git[0].note == "the git folder of this worktree"
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
    assert git[0].note == "the git folder of this submodule"


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
        import shutil
        shutil.rmtree(wt)
        wt.symlink_to(proj, target_is_directory=True)
    else:
        (wt / ".git").unlink()
        (wt / ".git").symlink_to(proj / ".git")
    (proj / ".git").write_text(f"gitdir: {entry}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []


def test_a_submodule_whose_worktree_names_another_folder_mounts_nothing(home):
    lib = _private_repo(home)
    proj = home / "src" / "proj"
    subprocess.run(["git", "config", "--file", str(lib / ".git" / "config"),
                    "core.worktree", str(home / "elsewhere")], check=True)
    (proj / ".git").write_text(f"gitdir: {lib / '.git'}\n")
    git, notes = _proj_git_mounts(home, proj)
    assert git == []


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
    if not settings.path_inside("/A", "/a"):
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
    plan = _plan(home, cwd=str(docs))
    assert any("~/Documents" in w and "macOS may ask" in w for w in plan.warnings)
    assert not _plan(home).warnings


def test_memory_warning(monkeypatch):
    assert settings.memory_warning("1024G") is not None
    assert settings.memory_warning("1G") is None
    # A 16 GB Mac: the 4G default is exactly a quarter, so it does not warn.
    pages = {"SC_PAGE_SIZE": 16384, "SC_PHYS_PAGES": (16 << 30) // 16384}
    monkeypatch.setattr(settings.os, "sysconf", lambda name: pages[name])
    assert settings.memory_warning("4G") is None
    assert settings.memory_warning("4097M") is not None


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


def test_seed_copies_once_then_adds_the_git_identity(home):
    (home / ".gitconfig").write_text("[user]\n\tname = Host Name\n\temail = host@example.com\n")
    (home / ".claude").mkdir()
    (home / ".claude" / "CLAUDE.md").write_text("rules")
    private = settings.private_home("claude-code")
    assert settings.seed_home(private, ["~/.claude/CLAUDE.md"]) == []
    assert (private / ".claude" / "CLAUDE.md").read_text() == "rules"
    (home / ".claude" / "CLAUDE.md").write_text("changed")
    settings.seed_home(private, ["~/.claude/CLAUDE.md"])
    assert (private / ".claude" / "CLAUDE.md").read_text() == "rules"

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


def test_seed_outside_home_is_refused_and_sensitive_warns(home, tmp_path):
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="not inside your home"):
        settings.seed_home(private, [str(tmp_path)])
    (home / ".npmrc").write_text("//registry/:_authToken=x")
    warns = settings.seed_home(private, ["~/.npmrc"])
    assert any("~/.npmrc" in w for w in warns)


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


def test_confine_refuses_a_private_home_outside_confined(home):
    from gmlx.container import confine
    private = settings.private_home("pi")
    for call in (lambda: confine.read_text(private / "x"),
                 lambda: confine.write_text(private / "x", "y"),
                 lambda: confine.exists(private / "x"),
                 lambda: confine.mkdirs(private / "d"),
                 lambda: confine.listdir(private)):
        with pytest.raises(confine.ConfinedError, match="private home"):
            call()
    confine.write_text(home / "ok.txt", "fine")          # the Mac's own files still work
    assert (home / "ok.txt").read_text() == "fine"


def test_seed_refuses_the_launch_data_folder(home):
    private = settings.private_home("pi")
    with pytest.raises(SettingsError, match="launch data folder"):
        settings.seed_home(private, ["~/.local"])
    with pytest.raises(SettingsError, match="launch data folder"):
        settings.seed_home(private, ["~/.local/share/gmlx/launch/omp"])


def test_a_relative_seed_is_read_from_home(home, monkeypatch):
    (home / "notes.md").write_text("n")
    monkeypatch.chdir(home / "src" / "proj")
    private = settings.private_home("pi")
    assert settings.seed_home(private, ["notes.md"]) == []
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
    assert len(out) == 1 and "is inside the read-write share" in out[0]
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


def test_broken_config_never_stops_the_launch(home):
    proj = home / "src" / "proj"
    cfg = _config(proj / "gmlx.yaml", "server: [unclosed\n")
    out = settings.server_config_warnings(cfg, _share(proj))
    assert any("could not check" in w for w in out)
    assert any("is inside the read-write share" in w for w in out)


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
