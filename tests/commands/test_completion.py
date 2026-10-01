#!/usr/bin/env python3
"""Shell-completion engine (`gmlx __complete`) + `gmlx completion zsh`.

CPU-only: every candidate is computed from argparse help, the verb table, and a
temp config - no model, no server, no shell.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import gmlx.commands.cli as cli
import gmlx.commands.completion as completion
import gmlx.serve.lifecycle as lifecycle


def _vals(lines):
    """The value column (before any tab) of candidate lines."""
    return [ln.split("\t", 1)[0] for ln in lines]


# Verb-level (first word) completion.

def test_verb_completion_lists_verbs_and_alias():
    vals = _vals(completion._complete([""]))
    for v in ("run", "chat", "serve", "launch", "list", "ps", "completion"):
        assert v in vals
    assert "ls" in vals                     # the list alias is offered too
    assert "__complete" not in vals         # the hidden helper never shows


def test_every_verb_has_a_description():
    # Drift guard: a new dispatchable verb must get a one-liner here.
    for v in cli._VERBS:
        assert completion._VERB_DESC.get(v), f"missing _VERB_DESC for {v!r}"


def test_verb_candidates_carry_descriptions():
    line = next(ln for ln in completion._complete([""]) if ln.startswith("run\t"))
    assert "generate" in line


# Flag completion (scraped from each verb's own --help).

def test_a_flag_description_is_its_first_sentence():
    """A description ends at a full stop, never where argparse wrapped the
    help, even when the sentence runs over two lines."""
    opts = {o: h for o, _, h in completion._verb_options("launch")}
    assert opts["--mount"] == "Share another folder with the container."
    assert opts["--config-only"] == ("Write the client's config and print the command "
                                     "instead of running it.")
    assert completion._first_sentence("Use e.g. this. Then that.") == "Use e.g. this."


def test_each_flag_of_a_launch_pair_describes_itself():
    opts = {o: h for o, _, h in completion._verb_options("launch")}
    assert opts["--container"] == ("Run the client in an Apple container, whatever the "
                                   "config says.")
    assert opts["--no-container"] == "Run the client on the Mac, whatever the config says."
    assert opts["--mount-cwd"] == "Share the current folder with the container."
    assert opts["--no-mount-cwd"] == "Do not share the current folder with the container."


def test_run_flag_completion():
    vals = _vals(completion._complete(["run", "--"]))
    for f in ("--max-tokens", "--temp", "--mmproj", "--speculative"):
        assert f in vals


def test_flag_help_survives_wrapping():
    # `-n LINES, --lines LINES` wraps its help onto the next line in argparse output;
    # the scraper must still attach it.
    lines = completion._complete(["logs", "--"])
    n = next(ln for ln in lines if ln.startswith("-n\t"))
    assert "history" in n


def test_service_borrows_serve_flags():
    vals = _vals(completion._complete(["service", "install", "--"]))
    assert "--host" in vals and "--port" in vals


# Value completion: files vs config models vs enums.

def test_path_flag_value_defers_to_files():
    lines = completion._complete(["run", "--config", ""])
    assert lines and lines[0] == "::files"


def test_non_path_flag_value_offers_nothing():
    # --temp takes a float we can't enumerate: no candidates, and crucially no files.
    assert completion._complete(["run", "--temp", ""]) == []


def test_choices_flag_value_completes_choices():
    assert completion._complete(["chat", "--thinking", ""]) == \
        ["on", "off", "adaptive"]
    assert completion._complete(["run", "--reasoning", ""]) == \
        ["show", "hide", "raw"]


def test_theme_flag_value_completes_themes():
    vals = _vals(completion._complete(["chat", "--theme", ""]))
    assert "dark" in vals and "light" in vals


def test_profile_flag_value_completes_intents():
    vals = _vals(completion._complete(["chat", "--profile", ""]))
    assert "coding" in vals and "reasoning-high" in vals


def _write_cfg(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(textwrap.dedent("""
        server:
          model_dirs: []
          assistants:
            helper:
              model: qwen-fast
        models:
          qwen-fast:
            path: /tmp/qwen.gguf
          gemma-vlm:
            path: /tmp/gemma.gguf
        aliases:
          q: qwen-fast
    """).strip())
    return str(cfg)


def test_model_positional_lists_config_ids_and_files(tmp_path):
    cfg = _write_cfg(tmp_path)
    lines = completion._complete(["run", "--config", cfg, ""])
    assert lines[0] == "::files"            # a path is always acceptable too
    vals = _vals(lines)
    assert "qwen-fast" in vals and "gemma-vlm" in vals
    assert "q" in vals                      # aliases included
    alias = next(ln for ln in lines if ln.startswith("q\t"))
    assert "alias -> qwen-fast" in alias
    assert "helper" in vals                 # served assistants included
    helper = next(ln for ln in lines if ln.startswith("helper\t"))
    assert "assistant -> qwen-fast" in helper


def test_launch_model_flag_lists_config_ids_without_files(tmp_path):
    cfg = _write_cfg(tmp_path)
    lines = completion._complete(
        ["launch", "opencode", "--config", cfg, "--model", ""])
    vals = _vals(lines)
    assert "::files" not in vals
    assert "qwen-fast" in vals and "gemma-vlm" in vals and "q" in vals


def test_model_positional_drops_off_once_model_given(tmp_path):
    cfg = _write_cfg(tmp_path)
    # A model is already in place; the next bare word is not a second model.
    assert completion._complete(["run", "--config", cfg, "qwen-fast", ""]) == []


def test_talk_positional_lists_model_ids_without_files(tmp_path):
    cfg = _write_cfg(tmp_path)
    lines = completion._complete(["talk", "--config", cfg, ""])
    assert "::files" not in lines           # a served id, never a path on disk
    vals = _vals(lines)
    assert "qwen-fast" in vals and "q" in vals
    # Once a model is chosen, no more positional candidates.
    assert completion._complete(["talk", "--config", cfg, "qwen-fast", ""]) == []


def test_ls_alias_canonicalizes_for_flags():
    vals = _vals(completion._complete(["ls", "--"]))
    assert "--config" in vals and "--json" in vals


# Positional value sources that aren't config models.

def test_launch_completes_harnesses_and_menubar():
    vals = _vals(completion._complete(["launch", ""]))
    for h in ("opencode", "claude-code", "menubar"):
        assert h in vals
    # Once a harness is chosen, no more positional candidates.
    assert completion._complete(["launch", "opencode", ""]) == []


def test_launch_labels_harnesses_by_kind():
    labels = dict(v.split("\t", 1)
                  for v in completion._complete(["launch", ""]))
    assert labels["pi"] == "coding agent"
    assert labels["goose"] == "agent runtime"
    assert labels["elia"] == "chat TUI"
    assert labels["dsh"] == "web app" and labels["open-webui"] == "web app"


def test_launch_dsh_profile_completes_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    (tmp_path / "profiles" / "tui").mkdir(parents=True)
    (tmp_path / "profiles" / "tui" / "package.json").write_text("{}")
    (tmp_path / "profiles" / "stray").mkdir()            # no manifest
    vals = _vals(completion._complete(
        ["launch", "dsh", "--dsh-profile", ""]))
    assert set(vals) == {"gmlx", "headless", "tui", "web"}


def test_launch_dsh_profile_completes_private_home_profiles_in_container_mode(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "host-dsh"))
    (tmp_path / "host-dsh" / "profiles" / "mac-only").mkdir(parents=True)
    (tmp_path / "host-dsh" / "profiles" / "mac-only" / "package.json").write_text("{}")
    guest = _guest_profiles()
    (guest / "boxed").mkdir(parents=True)
    (guest / "boxed" / "package.json").write_text("{}")
    vals = _vals(completion._complete(["launch", "dsh", "--container", "--dsh-profile", ""]))
    assert "boxed" in vals and "mac-only" not in vals
    vals = _vals(completion._complete(["launch", "dsh", "--dsh-profile", ""]))
    assert "mac-only" in vals and "boxed" not in vals


def test_private_home_profiles_never_follow_a_planted_link(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    mac = tmp_path / "mac-profiles"
    (mac / "secret-project").mkdir(parents=True)
    (mac / "secret-project" / "package.json").write_text("{}")
    guest = _guest_profiles()
    (guest / "boxed").mkdir(parents=True)
    (guest / "boxed" / "package.json").write_text("{}")
    (guest / "linked").symlink_to(mac / "secret-project", target_is_directory=True)
    vals = _vals(completion._complete(["launch", "dsh", "--container", "--dsh-profile", ""]))
    assert "boxed" in vals and "linked" not in vals
    guest.rename(guest.with_name("real"))
    guest.symlink_to(mac, target_is_directory=True)     # the whole folder is a link
    vals = _vals(completion._complete(["launch", "dsh", "--container", "--dsh-profile", ""]))
    assert "secret-project" not in vals


_HOSTILE = ("$(touch${IFS}PWNED)", "`touch PWNED2`", "a;touch PWNED3")


def _profile(root, name):
    (root / name).mkdir(parents=True)
    (root / name / "package.json").write_text("{}")


def _guest_profiles(words=()):
    """The dsh profiles in the private home that a container launch from
    the current folder uses for a profile of its own."""
    from gmlx.container import settings
    home = settings.private_home_path("dsh", completion._dsh_project(list(words)))
    return home / ".dsh" / "profiles"


def test_container_profiles_come_from_the_current_projects_home(tmp_path, monkeypatch):
    import tempfile

    from gmlx.container import settings
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    proj = Path(tempfile.mkdtemp(prefix="gc-", dir="/tmp"))
    try:
        monkeypatch.chdir(proj)
        project = settings.project_id(settings.canonical(str(proj)))
        _profile(settings.private_home_path("dsh", project) / ".dsh" / "profiles", "mine")
        _profile(settings.private_home_path("dsh") / ".dsh" / "profiles", "shared")
        vals = _vals(completion._complete(["launch", "dsh", "--container", "--dsh-profile", ""]))
        assert "mine" in vals and "shared" not in vals
        vals = _vals(completion._complete(
            ["launch", "dsh", "--no-mount-cwd", "--dsh-profile", ""]))
        assert "shared" in vals and "mine" not in vals
    finally:
        import shutil
        shutil.rmtree(proj, ignore_errors=True)


def test_container_profiles_follow_the_project_that_launch_keys(tmp_path, monkeypatch):
    """With mount_cwd false, a plain launch keys the default project, and
    --mount . or --mount-cwd keys the current folder's, as launch does."""
    import shutil
    import tempfile

    from gmlx.container import settings
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    cfg = tmp_path / ".config" / "gmlx" / "gmlx.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("launch:\n  container:\n    mount_cwd: false\n")
    proj = Path(tempfile.mkdtemp(prefix="gc-", dir="/tmp"))
    try:
        monkeypatch.chdir(proj)
        project = settings.project_id(settings.canonical(str(proj)))
        _profile(settings.private_home_path("dsh", project) / ".dsh" / "profiles", "mine")
        _profile(settings.private_home_path("dsh") / ".dsh" / "profiles", "shared")
        for flags, found in (([], "shared"), (["--mount", "."], "mine"),
                             (["--mount=."], "mine"), (["--mount-cwd"], "mine"),
                             (["--no-mount-cwd", "--mount", str(proj)], "mine"),
                             (["--mount", ".", "--no-mount-cwd"], "mine"),
                             (["--mount", str(tmp_path), "--no-mount-cwd"], "shared")):
            vals = _vals(completion._complete(
                ["launch", "dsh", "--container", *flags, "--dsh-profile", ""]))
            assert found in vals and {"mine", "shared"} - {found} - set(vals), flags
    finally:
        shutil.rmtree(proj, ignore_errors=True)


def _dsh_session(folder, *, alive=True):
    """Writes the record of a dsh session keyed to ``folder``, whose launch
    is this process, or a process that has ended. Gives its project id."""
    from gmlx.container import session, settings
    real = settings.canonical(str(folder)) if folder else None
    project = settings.project_id(real)
    owner = session.launch_owner()
    if not alive:
        owner["pid_start"] = (owner["pid_start"] or 0) + 1
    session.write_record("dsh", project, {
        "name": f"gmlx-dsh-{project}", "workdir": "/workspace", "project": real,
        "shares": [{"host": real, "guest": "/workspace", "readonly": False}] if real else [],
        "profile": "mine", **owner})
    return project


def test_container_profiles_follow_the_session_that_a_launch_joins(tmp_path, monkeypatch):
    """A launch from a folder that a running dsh session holds joins that
    session, so completion lists the profiles in that session's home."""
    import shutil
    import tempfile

    from gmlx.container import session, settings
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    cfg = tmp_path / ".config" / "gmlx" / "gmlx.yaml"
    cfg.parent.mkdir(parents=True)
    proj = Path(tempfile.mkdtemp(prefix="gc-", dir="/tmp"))
    (proj / "sub").mkdir()

    def found(*flags):
        vals = set(_vals(completion._complete(
            ["launch", "dsh", "--container", *flags, "--dsh-profile", ""])))
        return {"mine", "shared", "sub"} & vals

    try:
        project = settings.project_id(settings.canonical(str(proj)))
        _profile(settings.private_home_path("dsh", project) / ".dsh" / "profiles", "mine")
        _profile(settings.private_home_path("dsh") / ".dsh" / "profiles", "shared")
        sub = settings.project_id(settings.canonical(str(proj / "sub")))
        _profile(settings.private_home_path("dsh", sub) / ".dsh" / "profiles", "sub")
        # mount_cwd false: the session that --mount . started takes a plain
        # launch from its folder and from a folder inside it.
        cfg.write_text("launch:\n  container:\n    mount_cwd: false\n")
        _dsh_session(proj)
        for where in (proj, proj / "sub"):
            monkeypatch.chdir(where)
            assert found() == {"mine"}, where
            assert found("--no-mount-cwd") == {"shared"}, where
            assert found("--mount", str(tmp_path)) == {"shared"}, where
        # A launch keyed to a subfolder joins the session of its parent.
        cfg.write_text("launch:\n  container:\n    mount_cwd: true\n")
        assert found() == {"mine"}
        # A session whose launch has ended takes nothing.
        _dsh_session(proj, alive=False)
        assert found() == {"sub"}
        cfg.write_text("launch:\n  container:\n    mount_cwd: false\n")
        assert found() == {"shared"}
        # A running session of the default project comes first.
        _dsh_session(proj)
        _dsh_session(None)
        assert found() == {"shared"}
        session.remove_record("dsh", settings.PROJECT_DEFAULT)
        assert found() == {"mine"}
    finally:
        shutil.rmtree(proj, ignore_errors=True)


def test_launch_completes_remove_home():
    assert "--remove-home" in _vals(completion._complete(["launch", "pi", "--rem"]))


def test_dsh_profile_folders_with_shell_syntax_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "host-dsh"))
    host = tmp_path / "host-dsh" / "profiles"
    guest = _guest_profiles()
    for root in (host, guest):
        _profile(root, "fine-1.0")
        for name in _HOSTILE:
            _profile(root, name)
    for line in (["launch", "dsh", "--dsh-profile", ""],
                 ["launch", "dsh", "--container", "--dsh-profile", ""]):
        vals = _vals(completion._complete(line))
        assert "fine-1.0" in vals
        assert not set(vals) & set(_HOSTILE)


def test_cmd_complete_drops_unsafe_values_and_cleans_descriptions(monkeypatch, capsys):
    monkeypatch.setattr(completion, "_complete", lambda argv: [
        "::files", "ok-model\tdesc\x1bwith escape", *(f"{h}\tx" for h in _HOSTILE),
        "http://127.0.0.1:8080/v1\trunning server"])
    assert completion.cmd_complete(["run", ""]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "::files", "ok-model\tdesc with escape", "http://127.0.0.1:8080/v1\trunning server"]


def test_bash_script_never_expands_a_candidate(tmp_path):
    """Candidates reach bash as literal words, even ones that get past the
    Python filter, so completing a line runs nothing."""
    import shutil
    import subprocess
    bash = "/bin/bash" if shutil.which("/bin/bash") else shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    script = tmp_path / "gmlx.bash"
    script.write_text(completion._BASH_SCRIPT)
    driver = textwrap.dedent(f"""
        cd {tmp_path}
        gmlx() {{ printf '%s\\n' '$(touch PWNED)' '`touch PWNED2`' 'safe-one' 'other'; }}
        complete() {{ :; }}
        . {script}
        COMP_WORDS=(gmlx launch dsh --dsh-profile "")
        COMP_CWORD=4
        _gmlx
        printf '%s\\n' "${{COMPREPLY[@]}}"
        COMP_WORDS=(gmlx launch dsh --dsh-profile s)
        _gmlx
        printf 'prefix:%s\\n' "${{COMPREPLY[@]}}"
    """)
    done = subprocess.run([bash, "-c", driver], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert not list(tmp_path.glob("PWNED*"))
    lines = done.stdout.splitlines()
    # A candidate with a space or parentheses goes in quoted, as one word.
    assert r"\$\(touch\ PWNED\)" in lines and "safe-one" in lines
    assert [ln for ln in lines if ln.startswith("prefix:")] == ["prefix:safe-one"]


def test_a_model_id_with_a_space_or_parentheses_completes(tmp_path, monkeypatch, capsys):
    """zsh and fish quote such a value, and the bash script quotes it."""
    import shutil
    import subprocess
    cfg = tmp_path / "c.yaml"
    cfg.write_text("models:\n  qwen (fast):\n    path: /tmp/qwen.gguf\n"
                   "  plain:\n    path: /tmp/p.gguf\n")
    assert completion.cmd_complete(["launch", "pi", "--config", str(cfg), "--model",
                                    ""]) == 0
    assert _vals(capsys.readouterr().out.splitlines()) == ["qwen (fast)", "plain"]
    # A runfile value keeps the strict set.
    monkeypatch.setattr(completion, "_running_servers", lambda: [
        {"host": "evil host", "port": 1}, {"host": "127.0.0.1", "port": 8080}])
    assert _vals(completion._endpoint_candidates("HOST", "--host")) == ["127.0.0.1"]
    bash = "/bin/bash" if shutil.which("/bin/bash") else shutil.which("bash")
    if bash is None:
        return
    script = tmp_path / "gmlx.bash"
    script.write_text(completion._BASH_SCRIPT)
    driver = textwrap.dedent(f"""
        gmlx() {{ printf '%s\\t%s\\n' 'qwen (fast)' 'x.gguf' 'plain' 'p.gguf'; }}
        complete() {{ :; }}
        . {script}
        COMP_WORDS=(gmlx run q)
        COMP_CWORD=2
        _gmlx
        printf '%s\\n' "${{COMPREPLY[@]}}"
    """)
    done = subprocess.run([bash, "-c", driver], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [r"qwen\ \(fast\)"]


def test_bash_matches_and_passes_on_the_word_without_its_quotes(tmp_path):
    """COMP_WORDS holds each word as typed. A prefix typed with backslashes
    or in quotes still matches, and gmlx gets the words without quotes."""
    import shutil
    import subprocess
    bash = "/bin/bash" if shutil.which("/bin/bash") else shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    script = tmp_path / "gmlx.bash"
    script.write_text(completion._BASH_SCRIPT)
    driver = textwrap.dedent(f"""
        gmlx() {{ printf '%s\\n' "$@" > {tmp_path}/args
                 [[ -n $FILES ]] && echo ::files
                 printf '%s\\t%s\\n' 'qwen (fast)' a 'qwen (slow)' b; }}
        complete() {{ :; }}
        . {script}
        show() {{ COMP_CWORD=$(( ${{#COMP_WORDS[@]}} - 1 )); _gmlx
                 printf '%s|' "${{COMPREPLY[@]}}"; echo; }}
        COMP_WORDS=(gmlx run 'qwen\\ \\(f'); show
        COMP_WORDS=(gmlx run '"qwen (s'); show
        COMP_WORDS=(gmlx run "'qwen (f"); show
        COMP_WORDS=(gmlx run 'qwen\\ "(f'); show
        COMP_WORDS=(gmlx run 'qwen\\ '); show
        COMP_WORDS=(gmlx run --config 'my\\ dir/"c.yaml"' 'q'); show
        cat {tmp_path}/args
        compopt() {{ :; }}      # bash 4 and later quote file names themselves
        FILES=1
        COMP_WORDS=(gmlx run 'qwen\\ \\(f'); show
    """)
    done = subprocess.run([bash, "-c", driver], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        r"qwen\ \(fast\)|",
        "qwen (slow)|",                     # in an open quote, the quote keeps one word
        "qwen (fast)|",
        "(fast)|",                          # only the quoted part is replaced
        r"qwen\ \(fast\)|qwen\ \(slow\)|",
        r"qwen\ \(fast\)|qwen\ \(slow\)|",
        "__complete", "run", "--config", "my dir/c.yaml", "q",
        "qwen (fast)|"]


@pytest.mark.parametrize("flag", ["--shell", "--rebuild", "--mount=/x", "--no-mount-cwd",
                                  "--image", "--network"])
def test_container_only_flags_select_private_home_profiles(tmp_path, monkeypatch, flag):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "host-dsh"))
    _profile(tmp_path / "host-dsh" / "profiles", "mac-only")
    _profile(_guest_profiles([flag]), "boxed")
    vals = _vals(completion._complete(["launch", "dsh", flag, "--dsh-profile", ""]))
    assert "boxed" in vals and "mac-only" not in vals
    vals = _vals(completion._complete(
        ["launch", "dsh", "--no-container", flag, "--dsh-profile", ""]))
    assert "mac-only" in vals and "boxed" not in vals


def test_launch_offers_no_gmlx_flags_after_a_bare_separator():
    assert completion._complete(["launch", "pi", "--", "-"]) == ["::files"]
    assert completion._complete(["launch", "pi", "--", "--model", ""]) == ["::files"]
    assert completion._complete(["launch", "pi", "--", ""]) == ["::files"]


def test_launch_container_flags_complete():
    assert completion._complete(["launch", "pi", "--mount", ""]) == ["::files"]
    assert completion._complete(["launch", "pi", "--network", ""]) == ["default", "none"]


def test_service_completes_actions():
    vals = _vals(completion._complete(["service", ""]))
    assert vals == ["install", "uninstall", "status"] or set(vals) == {
        "install", "uninstall", "status"}


def test_validate_positional_defers_to_files():
    assert completion._complete(["validate", ""]) == ["::files"]


def test_unknown_verb_yields_nothing():
    assert completion._complete(["frobnicate", ""]) == []


# Live endpoint completion: --host/--port/--url/--base-url from running servers.

_FAKE_RUNS = [
    {"host": "127.0.0.1", "port": 8080,
     "url": "http://127.0.0.1:8080", "managed_by": "detach"},
    {"host": "127.0.0.1", "port": 8081,
     "url": "http://127.0.0.1:8081", "managed_by": "detach"},
    {"host": "0.0.0.0", "port": 9090,
     "url": "http://0.0.0.0:9090", "managed_by": "launchd"},
]


def test_port_value_completes_running_ports(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["serve", "--port", ""]))
    assert vals == ["9090", "8081", "8080"] or set(vals) == {"8080", "8081", "9090"}
    # The host + how-it's-managed ride along as the description.
    line = next(ln for ln in completion._complete(["stop", "--port", ""])
                if ln.startswith("9090\t"))
    assert "0.0.0.0" in line and "launchd" in line


def test_host_value_completes_and_dedupes(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["status", "--host", ""]))
    assert set(vals) == {"127.0.0.1", "0.0.0.0"}   # three runs, two distinct hosts


def test_url_value_completes_running_urls(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["ps", "--url", ""]))
    assert "http://0.0.0.0:9090" in vals
    assert "http://127.0.0.1:8080" in vals


def test_base_url_value_appends_v1(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: list(_FAKE_RUNS))
    vals = _vals(completion._complete(["launch", "--base-url", ""]))
    assert "http://127.0.0.1:8080/v1" in vals     # base-url wants the /v1 suffix
    assert "http://127.0.0.1:8080" not in vals


def test_endpoint_value_empty_with_no_servers(monkeypatch):
    monkeypatch.setattr(lifecycle, "list_runs", lambda: [])
    # Nothing running -> no candidates, and crucially no spurious ::files.
    assert completion._complete(["serve", "--port", ""]) == []
    assert completion._complete(["ps", "--url", ""]) == []


# Robustness: completion must never raise.

def test_complete_swallows_bad_config(tmp_path, capsys):
    bad = tmp_path / "broken.yaml"
    bad.write_text("models: [this is not: valid: yaml")
    assert completion.cmd_complete(["run", "--config", str(bad), ""]) == 0
    # Whatever it prints, it exits cleanly (no traceback).
    assert "Traceback" not in capsys.readouterr().err


def test_cmd_complete_prints_lines(capsys):
    assert completion.cmd_complete([""]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "serve" in out


# `completion zsh` script emission.

def test_completion_zsh_emits_script(capsys):
    assert completion.cmd_completion(["zsh"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("#compdef gmlx")  # the command name
    assert "gmlx __complete" in out         # the live callback
    assert "funcstack[1] == _gmlx" in out  # dual eval/fpath idiom
    assert "compdef _gmlx gmlx" in out


def test_completion_bash_emits_script(capsys):
    assert completion.cmd_completion(["bash"]) == 0
    out = capsys.readouterr().out
    assert "complete -F _gmlx gmlx" in out  # the command name
    assert "gmlx __complete" in out         # same live callback as zsh
    assert "COMPREPLY" in out


def test_completion_fish_emits_script(capsys):
    assert completion.cmd_completion(["fish"]) == 0
    out = capsys.readouterr().out
    assert "complete -c gmlx" in out        # the command name
    assert "gmlx __complete" in out         # same live callback as zsh/bash
    assert "__fish_complete_path" in out     # ::files is handled in fish too


def test_completion_no_shell_prints_help(capsys):
    assert completion.cmd_completion([]) == 0
    assert "completion script" in capsys.readouterr().out.lower()


def test_completion_rejects_unknown_shell(capsys):
    with pytest.raises(SystemExit):
        completion.cmd_completion(["powershell"])  # not implemented; argparse rejects
    assert "powershell" in capsys.readouterr().err


# Umbrella routing.

def test_umbrella_routes_hidden_complete(capsys):
    assert cli.umbrella_main(["__complete", ""]) == 0
    assert "run" in capsys.readouterr().out


def test_umbrella_routes_completion_verb(capsys):
    assert cli.umbrella_main(["completion", "zsh"]) == 0
    assert capsys.readouterr().out.startswith("#compdef gmlx")


def test_completion_is_a_known_verb():
    assert "completion" in cli._VERBS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
