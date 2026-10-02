"""gmlx/serve/media_programs.py and gmlx/serve/programs.py: the server runs
the ffmpeg and ffprobe on its PATH, but never one from a folder that a
container client can write: a folder that a session shared read-write, the
private homes, or the folder that a relative PATH entry names."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import types

import numpy as np
import pytest

from gmlx.container import settings
from gmlx.container.state import data_path
from gmlx.safe_path import canonical
from gmlx.serve import media_programs, media_sinks, programs, server, stt, tts

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
# What the fake ffmpeg prints: two 16-bit samples.
PCM = b"abcd"
SAMPLES = np.frombuffer(PCM, dtype=np.int16)


def _program(folder, name: str, log, body: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f'#!/bin/sh\necho "$0" >> "{log}"\ncat > /dev/null\n{body}\n')
    path.chmod(0o755)


def _planted(folder, log) -> None:
    """A client's ffmpeg and ffprobe, which log their run and fail."""
    for name in ("ffmpeg", "ffprobe"):
        _program(folder, name, log, "exit 1")


def _working(folder, log) -> None:
    """An ffprobe and an ffmpeg that log their run and work."""
    _program(folder, "ffprobe", log,
             """echo '{"streams": [{"sample_rate": "16000", "channels": 1}]}'""")
    _program(folder, "ffmpeg", log, f"printf '{PCM.decode()}'")


def _shared(*folders) -> None:
    """Record ``folders`` in launch's history of read-write shares."""
    path = data_path() / "shared.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text())["shared"] if path.exists() else []
    path.write_text(json.dumps({"shared": [*map(canonical, folders), *old],
                                "worktrees": []}))


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    """The Homebrew and system folders, empty, in place of the real ones."""
    folder = tmp_path / "fixed" / "bin"
    folder.mkdir(parents=True)
    monkeypatch.setattr(settings, "SYSTEM_PATH", str(folder))
    return folder


@pytest.fixture
def ran(tmp_path, monkeypatch, fixed):
    """Plants ffmpeg and ffprobe in a shared project's .venv/bin first on
    PATH, and puts working ones in a folder after it. Gives the programs
    that ran."""
    log = tmp_path / "ran.txt"
    share = tmp_path / "share"
    _planted(share / ".venv" / "bin", log)
    _working(tmp_path / "tools" / "bin", log)
    _shared(share)
    monkeypatch.setenv("PATH", f"{share}/.venv/bin:{tmp_path}/tools/bin:/usr/bin:/bin")
    media_sinks.install()

    def programs_ran() -> list[str]:
        lines = log.read_text().splitlines() if log.exists() else []
        return [line.replace(str(tmp_path), "") for line in lines]
    yield programs_ran
    media_sinks.uninstall()


def _whisper(monkeypatch) -> list:
    heard = []
    mod = types.ModuleType("mlx_whisper")

    def transcribe(audio, *, path_or_hf_repo, temperature, **decode_options):
        heard.append(audio)
        return {"text": "hi", "segments": []}

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)

    @contextlib.contextmanager
    def online(_ref):
        yield
    monkeypatch.setattr(stt, "offline_resolve", online)
    return heard


def test_request_audio_is_decoded_with_the_programs_on_path_outside_a_share(ran):
    audio_io = importlib.import_module("mlx_audio.audio_io")
    samples, rate = audio_io.read(MP4, dtype="int16")
    assert rate == 16000 and samples.tolist() == SAMPLES.tolist()
    assert ran() == ["/tools/bin/ffprobe", "/tools/bin/ffmpeg"]


def test_speech_is_encoded_with_the_ffmpeg_on_path_outside_a_share(ran):
    audio = np.zeros(2400, dtype=np.float32)
    assert tts.encode_audio(audio, 24000, "mp3") == PCM
    assert tts.encode_audio(audio, 24000, "opus") == PCM
    assert ran() == ["/tools/bin/ffmpeg", "/tools/bin/ffmpeg"]


def test_an_upload_is_decoded_with_the_ffmpeg_on_path_outside_a_share(ran, monkeypatch):
    heard = _whisper(monkeypatch)
    content, _ = stt.run_transcription(b"x", filename="a.m4a", configured_model="whisper-1")
    assert content == {"text": "hi"}
    (audio,) = heard
    assert audio.dtype == np.float32 and audio.tolist() == (SAMPLES / 32768.0).tolist()
    assert ran() == ["/tools/bin/ffmpeg"]


@pytest.mark.filterwarnings("ignore:ffmpeg is required:RuntimeWarning")
def test_a_missing_program_names_the_skipped_share_and_the_step(ran, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", f"{tmp_path}/share/.venv/bin:/usr/bin:/bin")
    share = canonical(tmp_path / "share")
    want = (r"The gmlx server finds no ffmpeg on its PATH or in .*fixed/bin\. It does not "
            rf"look in .*share/\.venv/bin, because that PATH entry lies in {share}, a folder "
            r"that a container session shared read-write\. Install it with `brew install "
            r"ffmpeg`, or start the server from a shell whose PATH holds your ffmpeg\.")
    audio_io = importlib.import_module("mlx_audio.audio_io")
    with pytest.raises(media_programs.ProgramMissing, match=want):
        audio_io.read(MP4)
    with pytest.raises(media_programs.ProgramMissing, match=want):
        tts.encode_audio(np.zeros(4, dtype=np.float32), 24000, "mp3")
    _whisper(monkeypatch)
    with pytest.raises(RuntimeError, match="transcription failed: " + want):
        stt.run_transcription(b"x", filename="a.m4a", configured_model="whisper-1")
    assert ran() == []


def test_an_ffmpeg_from_another_package_manager_on_path_is_found(tmp_path, monkeypatch, fixed):
    """A MacPorts, Nix or conda ffmpeg on PATH works, with no Homebrew."""
    ports = tmp_path / "opt" / "local" / "bin"
    _working(ports, tmp_path / "ran.txt")
    monkeypatch.setenv("PATH", f"{ports}:/usr/bin:/bin")
    assert media_programs.find("ffmpeg") == str(ports / "ffmpeg")
    assert media_programs.program("ffprobe") == str(ports / "ffprobe")
    assert media_programs.problem("ffmpeg") is None


def test_a_relative_path_entry_is_never_searched(tmp_path, monkeypatch, fixed):
    """An empty entry or . names the folder the server runs in, which is
    often a project that the next launch shares."""
    project = tmp_path / "project"
    _working(project, tmp_path / "ran.txt")
    (project / "bin").mkdir()
    monkeypatch.chdir(project)
    for path, shown in ((".:/usr/bin", "."), (":/usr/bin", "the empty PATH entry"),
                        ("/usr/bin:", "the empty PATH entry"), ("bin/..:/usr/bin", "bin/..")):
        monkeypatch.setenv("PATH", path)
        assert media_programs.find("ffmpeg") is None, path
        found = programs.search()
        assert all(os.path.isabs(f) for f in found.folders)
        kind = "empty" if "empty" in shown else "relative"
        assert (f"It does not look in {shown}, because that PATH entry is {kind}, so it names "
                "the folder that the server runs in.") in (media_programs.problem("ffmpeg") or "")


def test_a_link_that_leads_into_a_share_is_refused(tmp_path, monkeypatch, fixed):
    """~/bin passes the folder check, but its ffmpeg is a link into a
    shared project, so the guest writes the program that would run."""
    log = tmp_path / "ran.txt"
    share = tmp_path / "proj"
    _planted(share / "tools", log)
    links = tmp_path / "home" / "bin"
    links.mkdir(parents=True)
    (links / "ffmpeg").symlink_to(share / "tools" / "ffmpeg")
    _working(tmp_path / "tools" / "bin", log)
    _shared(share)
    monkeypatch.setenv("PATH", f"{links}:{tmp_path}/tools/bin:/usr/bin:/bin")
    assert media_programs.find("ffmpeg") is None
    real = canonical(share / "tools" / "ffmpeg")
    want = (f"The gmlx server will not run {links}/ffmpeg, because it leads to {real}, in "
            f"{canonical(share)}, a folder that a container session shared read-write. A "
            "container client could have written that file. Remove that file, so that the "
            "server finds another one, or install ffmpeg with `brew install ffmpeg`.")
    assert media_programs.problem("ffmpeg") == want
    with pytest.raises(media_programs.ProgramRefused) as err:
        media_programs.program("ffmpeg")
    assert str(err.value) == want
    assert not log.exists()


def test_the_private_homes_and_a_link_into_a_share_are_skipped(tmp_path, monkeypatch, fixed):
    homes = data_path() / "homes" / "pi" / "bin"
    share = tmp_path / "share"
    _planted(homes, tmp_path / "ran.txt")
    _planted(share / "bin", tmp_path / "ran.txt")
    (tmp_path / "linked").symlink_to(share / "bin")
    _shared(share)
    monkeypatch.setenv("PATH", f"{homes}:{tmp_path}/linked:/usr/bin:/bin")
    found = programs.search()
    assert found.folders == ("/usr/bin", "/bin", str(fixed))
    reasons = dict(found.skipped)
    assert reasons[str(homes)].startswith(f"lies in {canonical(data_path())}, where launch "
                                          "keeps the private homes")
    assert reasons[f"{tmp_path}/linked"] == (
        f"leads to {canonical(share / 'bin')}, in {canonical(share)}, a folder that a "
        "container session shared read-write")
    assert media_programs.find("ffmpeg") is None


def test_a_skipped_link_does_not_hide_a_later_entry_with_the_same_real_path(
        tmp_path, monkeypatch, fixed):
    """A link in a share that leads to a safe folder is skipped, and the
    safe folder, later on PATH, is still searched."""
    tools, share = tmp_path / "tools", tmp_path / "share"
    _working(tools, tmp_path / "ran.txt")
    share.mkdir()
    (share / "sysbin").symlink_to(tools)
    _shared(share)
    monkeypatch.setenv("PATH", f"{share}/sysbin:{tools}:/usr/bin:/bin")
    found = programs.search()
    assert found.folders == (str(tools), "/usr/bin", "/bin", str(fixed))
    assert [entry for entry, _why in found.skipped] == [f"{share}/sysbin"]
    assert media_programs.find("ffmpeg") == str(tools / "ffmpeg")


def test_a_fixed_folder_off_path_is_searched_last(tmp_path, monkeypatch, fixed):
    """A login item's PATH leaves out /opt/homebrew/bin, so the server
    still searches the Homebrew and system folders after its PATH."""
    _working(fixed, tmp_path / "ran.txt")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert media_programs.find("ffmpeg") == str(fixed / "ffmpeg")
    assert programs.search().added == (str(fixed),)


def test_a_shared_fixed_folder_and_a_shared_installation_are_refused(tmp_path, monkeypatch):
    """A read-write share of /usr/local, or of Homebrew's Cellar where
    /opt/homebrew/bin/ffmpeg leads, lets the guest write the ffmpeg."""
    log = tmp_path / "ran.txt"
    local, brew = tmp_path / "local", tmp_path / "brew"
    _planted(local / "bin", log)
    _planted(brew / "Cellar" / "ffmpeg" / "bin", log)
    (brew / "bin").mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (brew / "bin" / name).symlink_to(brew / "Cellar" / "ffmpeg" / "bin" / name)
    monkeypatch.setattr(settings, "SYSTEM_PATH", f"{local}/bin:{brew}/bin:/usr/bin:/bin")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert media_programs.find("ffmpeg") == str(local / "bin" / "ffmpeg")
    _shared(local)
    assert media_programs.find("ffmpeg") == str(brew / "bin" / "ffmpeg")
    _shared(brew / "Cellar")
    assert media_programs.find("ffmpeg") is None
    assert (media_programs.problem("ffmpeg") or "").startswith(
        f"The gmlx server will not run {brew}/bin/ffmpeg, because it leads to "
        f"{canonical(brew / 'Cellar' / 'ffmpeg' / 'bin' / 'ffmpeg')}, in "
        f"{canonical(brew / 'Cellar')}, a folder that a container session")
    assert not log.exists()


def test_the_server_log_names_the_ffmpeg_and_each_skipped_entry_once(ran, capsys, tmp_path):
    programs._logged.clear()
    media_programs.log_programs(types.SimpleNamespace(stt="whisper", tts=None))
    media_programs.program("ffmpeg")
    media_programs.program("ffprobe")
    assert capsys.readouterr().err.splitlines() == [
        f"[server] ffmpeg: {tmp_path}/tools/bin/ffmpeg",
        f"[server] no program runs from the PATH entry {tmp_path}/share/.venv/bin, because "
        f"it lies in {canonical(tmp_path / 'share')}, a folder that a container session "
        "shared read-write.",
        f"[server] ffprobe: {tmp_path}/tools/bin/ffprobe",
    ]


def test_the_server_log_names_a_missing_ffmpeg(tmp_path, monkeypatch, fixed, capsys):
    """The start lines give the install step only on a server that serves
    transcription or speech. A request that needs ffmpeg writes it later."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    programs._logged.clear()
    media_programs.log_programs(types.SimpleNamespace(stt=None, tts=None))
    assert capsys.readouterr().err.splitlines() == [
        f"[server] ffmpeg: none. The gmlx server finds no ffmpeg on its PATH or in {fixed}.",
        f"[server] ffprobe: none. The gmlx server finds no ffprobe on its PATH or in {fixed}.",
    ]
    with pytest.raises(media_programs.ProgramMissing):
        media_programs.program("ffmpeg")
    assert capsys.readouterr().err.splitlines() == [
        f"[server] ffmpeg: none. The gmlx server finds no ffmpeg on its PATH or in {fixed}. "
        "Install it with `brew install ffmpeg`, or start the server from a shell whose PATH "
        "holds your ffmpeg."]
    programs._logged.clear()
    media_programs.log_programs(types.SimpleNamespace(stt=None, tts="kokoro"))
    assert "Install it with `brew install ffmpeg`" in capsys.readouterr().err


def test_a_share_that_a_later_session_adds_counts_at_the_next_lookup(tmp_path, monkeypatch,
                                                                      fixed):
    tools = tmp_path / "tools"
    _working(tools, tmp_path / "ran.txt")
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
    assert media_programs.find("ffmpeg") == str(tools / "ffmpeg")
    _shared(tools)
    assert media_programs.find("ffmpeg") is None


def test_the_install_hints_name_the_path_of_the_server(monkeypatch):
    for name, load in (("mlx_whisper", stt.import_mlx_whisper),
                       ("mlx_audio", tts.import_mlx_audio)):
        monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(ImportError) as err:
            load()
        assert ("ffmpeg on the PATH of the server. Install it with `brew install ffmpeg`."
                in str(err.value))
        assert "/opt/homebrew/bin" not in str(err.value)


def test_the_serve_help_names_the_path_of_the_server_for_ffmpeg():
    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    helps = {action.dest: action.help or "" for action in ap._actions}
    for dest in ("stt", "tts"):
        assert "ffmpeg on the server's PATH" in helps[dest], helps[dest]
        assert "/opt/homebrew/bin" not in helps[dest]
